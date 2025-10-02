#!/usr/bin/env python3
"""
DeepONet Auto-Scaling System

Intelligent auto-scaling system for DeepONet deployments including:
- Load-based scaling decisions
- Resource utilization monitoring
- Predictive scaling
- Multi-metric scaling policies
- Integration with Kubernetes HPA and VPA
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
import asyncio
from datetime import datetime, timedelta

import numpy as np
import torch
from kubernetes import client, config as k8s_config
from prometheus_api_client import PrometheusConnect


logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class MetricType(Enum):
    """Supported metric types."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    GPU_UTILIZATION = "gpu_utilization"
    GPU_MEMORY = "gpu_memory"


@dataclass
class ScalingMetric:
    """Configuration for a scaling metric."""
    metric_type: MetricType
    target_value: float
    min_value: float = 0.0
    max_value: float = 100.0
    weight: float = 1.0
    aggregation_window: int = 300  # seconds
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    enabled: bool = True


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    metrics: List[ScalingMetric] = field(default_factory=list)
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    prediction_window: int = 900  # seconds for predictive scaling
    enabled: bool = True


@dataclass
class ScalingDecision:
    """Scaling decision with context."""
    action: ScalingAction
    current_replicas: int
    target_replicas: int
    reasoning: str
    confidence: float
    metrics: Dict[str, float]
    timestamp: datetime


class MetricCollector:
    """Collects metrics for scaling decisions."""

    def __init__(self, prometheus_url: Optional[str] = None):
        self.prometheus_url = prometheus_url or "http://localhost:9090"
        self.prometheus = None
        self.metric_history: Dict[MetricType, deque] = {}
        self.history_lock = threading.Lock()

        # Initialize metric history
        for metric_type in MetricType:
            self.metric_history[metric_type] = deque(maxlen=1000)

        if prometheus_url:
            try:
                self.prometheus = PrometheusConnect(url=prometheus_url)
                logger.info(f"Connected to Prometheus at {prometheus_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Prometheus: {e}")

    def collect_metric(self, metric_type: MetricType, namespace: str = "deeponet") -> Optional[float]:
        """Collect a specific metric."""
        try:
            if self.prometheus:
                query = self._get_prometheus_query(metric_type, namespace)
                result = self.prometheus.get_current_metric_value(metric_name=query)

                if result and len(result) > 0:
                    value = float(result[0]['value'][1])

                    # Store in history
                    with self.history_lock:
                        self.metric_history[metric_type].append({
                            'timestamp': time.time(),
                            'value': value
                        })

                    return value
            else:
                # Fallback to system metrics
                return self._get_system_metric(metric_type)

        except Exception as e:
            logger.error(f"Failed to collect metric {metric_type}: {e}")
            return None

    def _get_prometheus_query(self, metric_type: MetricType, namespace: str) -> str:
        """Get Prometheus query for metric type."""
        queries = {
            MetricType.CPU_UTILIZATION: f'avg(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])) * 100',
            MetricType.MEMORY_UTILIZATION: f'avg(container_memory_usage_bytes{{namespace="{namespace}"}}) / avg(container_spec_memory_limit_bytes{{namespace="{namespace}"}}) * 100',
            MetricType.REQUEST_RATE: f'sum(rate(deeponet_requests_total{{namespace="{namespace}"}}[5m]))',
            MetricType.RESPONSE_TIME: f'histogram_quantile(0.95, rate(deeponet_request_duration_seconds_bucket{{namespace="{namespace}"}}[5m]))',
            MetricType.QUEUE_LENGTH: f'sum(deeponet_queue_length{{namespace="{namespace}"}})',
            MetricType.ERROR_RATE: f'sum(rate(deeponet_errors_total{{namespace="{namespace}"}}[5m])) / sum(rate(deeponet_requests_total{{namespace="{namespace}"}}[5m])) * 100',
            MetricType.THROUGHPUT: f'sum(rate(deeponet_requests_total{{namespace="{namespace}"}}[5m]))',
            MetricType.GPU_UTILIZATION: f'avg(DCGM_FI_DEV_GPU_UTIL{{namespace="{namespace}"}})',
            MetricType.GPU_MEMORY: f'avg(DCGM_FI_DEV_FB_USED{{namespace="{namespace}"}}) / avg(DCGM_FI_DEV_FB_TOTAL{{namespace="{namespace}"}}) * 100'
        }
        return queries.get(metric_type, "up")

    def _get_system_metric(self, metric_type: MetricType) -> float:
        """Get system metric as fallback."""
        try:
            if metric_type == MetricType.CPU_UTILIZATION:
                import psutil
                return psutil.cpu_percent(interval=1)
            elif metric_type == MetricType.MEMORY_UTILIZATION:
                import psutil
                return psutil.virtual_memory().percent
            elif metric_type == MetricType.GPU_UTILIZATION:
                if torch.cuda.is_available():
                    return torch.cuda.utilization()
                return 0.0
            elif metric_type == MetricType.GPU_MEMORY:
                if torch.cuda.is_available():
                    memory_info = torch.cuda.memory_stats()
                    allocated = memory_info.get('allocated_bytes.all.current', 0)
                    reserved = memory_info.get('reserved_bytes.all.current', 0)
                    return (allocated / reserved * 100) if reserved > 0 else 0.0
                return 0.0
            else:
                return 0.0  # Default value

        except Exception as e:
            logger.error(f"Failed to get system metric {metric_type}: {e}")
            return 0.0

    def get_metric_history(self, metric_type: MetricType, window_seconds: int = 300) -> List[float]:
        """Get metric history within time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self.history_lock:
            history = self.metric_history.get(metric_type, deque())
            recent_values = [
                entry['value'] for entry in history
                if entry['timestamp'] >= cutoff_time
            ]

        return recent_values

    def get_aggregated_metric(self, metric_type: MetricType, window_seconds: int = 300,
                            aggregation: str = "avg") -> Optional[float]:
        """Get aggregated metric value."""
        values = self.get_metric_history(metric_type, window_seconds)

        if not values:
            return None

        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "p95":
            return np.percentile(values, 95)
        elif aggregation == "p99":
            return np.percentile(values, 99)
        else:
            return statistics.mean(values)


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""

    def __init__(self, prediction_window: int = 900):
        self.prediction_window = prediction_window
        self.patterns: Dict[str, List[Tuple[datetime, float]]] = {}

    def train_predictor(self, metric_type: MetricType, metric_collector: MetricCollector):
        """Train predictor with historical data."""
        history = metric_collector.get_metric_history(
            metric_type, window_seconds=self.prediction_window * 4
        )

        if len(history) < 10:
            return

        # Simple pattern recognition (can be enhanced with ML models)
        pattern_key = f"{metric_type.value}_pattern"
        current_time = datetime.now()

        # Store patterns by time of day and day of week
        time_patterns = []
        for i in range(len(history) - 1):
            time_patterns.append((current_time - timedelta(seconds=i), history[i]))

        self.patterns[pattern_key] = time_patterns

    def predict_load(self, metric_type: MetricType, future_seconds: int = 300) -> Optional[float]:
        """Predict future load based on historical patterns."""
        pattern_key = f"{metric_type.value}_pattern"

        if pattern_key not in self.patterns or not self.patterns[pattern_key]:
            return None

        current_time = datetime.now()
        future_time = current_time + timedelta(seconds=future_seconds)

        # Find similar time patterns
        similar_patterns = []
        for timestamp, value in self.patterns[pattern_key]:
            if (timestamp.hour == future_time.hour and
                timestamp.weekday() == future_time.weekday()):
                similar_patterns.append(value)

        if similar_patterns:
            return statistics.mean(similar_patterns)

        return None


class KubernetesScaler:
    """Kubernetes integration for scaling operations."""

    def __init__(self, namespace: str = "deeponet"):
        self.namespace = namespace
        self.apps_v1 = None
        self.autoscaling_v1 = None

        try:
            # Load Kubernetes configuration
            try:
                k8s_config.load_incluster_config()
            except:
                k8s_config.load_kube_config()

            self.apps_v1 = client.AppsV1Api()
            self.autoscaling_v1 = client.AutoscalingV1Api()
            logger.info("Connected to Kubernetes API")

        except Exception as e:
            logger.warning(f"Failed to connect to Kubernetes: {e}")

    def get_current_replicas(self, deployment_name: str) -> int:
        """Get current number of replicas."""
        if not self.apps_v1:
            return 1

        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas or 1

        except Exception as e:
            logger.error(f"Failed to get current replicas: {e}")
            return 1

    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale deployment to target replicas."""
        if not self.apps_v1:
            return False

        try:
            # Update deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )

            deployment.spec.replicas = replicas

            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )

            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True

        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False

    def create_hpa(self, deployment_name: str, min_replicas: int = 1,
                   max_replicas: int = 10, target_cpu: int = 70) -> bool:
        """Create Horizontal Pod Autoscaler."""
        if not self.autoscaling_v1:
            return False

        try:
            hpa = client.V1HorizontalPodAutoscaler(
                metadata=client.V1ObjectMeta(name=f"{deployment_name}-hpa"),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    target_cpu_utilization_percentage=target_cpu
                )
            )

            self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )

            logger.info(f"Created HPA for {deployment_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create HPA: {e}")
            return False


class DeepONetAutoScaler:
    """Main auto-scaling controller for DeepONet."""

    def __init__(self, deployment_name: str = "deeponet-api",
                 namespace: str = "deeponet",
                 prometheus_url: Optional[str] = None):
        self.deployment_name = deployment_name
        self.namespace = namespace

        self.metric_collector = MetricCollector(prometheus_url)
        self.predictive_scaler = PredictiveScaler()
        self.k8s_scaler = KubernetesScaler(namespace)

        self.policies: Dict[str, ScalingPolicy] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.last_scale_time = {
            ScalingAction.SCALE_UP: 0,
            ScalingAction.SCALE_DOWN: 0
        }

        self.running = False
        self.scaling_thread = None

    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add a scaling policy."""
        self.policies[policy.name] = policy
        logger.info(f"Added scaling policy: {policy.name}")

    def remove_scaling_policy(self, policy_name: str):
        """Remove a scaling policy."""
        if policy_name in self.policies:
            del self.policies[policy_name]
            logger.info(f"Removed scaling policy: {policy_name}")

    def start_auto_scaling(self, check_interval: int = 30):
        """Start auto-scaling monitoring."""
        if self.running:
            return

        self.running = True
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            args=(check_interval,)
        )
        self.scaling_thread.start()
        logger.info("Started auto-scaling")

    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join()
        logger.info("Stopped auto-scaling")

    def _scaling_loop(self, check_interval: int):
        """Main scaling loop."""
        while self.running:
            try:
                self._evaluate_scaling()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(check_interval)

    def _evaluate_scaling(self):
        """Evaluate whether scaling is needed."""
        current_replicas = self.k8s_scaler.get_current_replicas(self.deployment_name)

        # Collect metrics for all policies
        policy_decisions = []

        for policy_name, policy in self.policies.items():
            if not policy.enabled:
                continue

            decision = self._evaluate_policy(policy, current_replicas)
            if decision.action != ScalingAction.NO_ACTION:
                policy_decisions.append((policy, decision))

        # Make final scaling decision
        if policy_decisions:
            final_decision = self._make_scaling_decision(policy_decisions, current_replicas)

            if final_decision.action != ScalingAction.NO_ACTION:
                self._execute_scaling_decision(final_decision)

    def _evaluate_policy(self, policy: ScalingPolicy, current_replicas: int) -> ScalingDecision:
        """Evaluate a single scaling policy."""
        metric_scores = {}
        scale_up_votes = 0
        scale_down_votes = 0
        reasoning_parts = []

        # Collect and evaluate metrics
        for metric in policy.metrics:
            if not metric.enabled:
                continue

            current_value = self.metric_collector.get_aggregated_metric(
                metric.metric_type, metric.aggregation_window
            )

            if current_value is None:
                continue

            metric_scores[metric.metric_type.value] = current_value

            # Check thresholds
            if current_value > metric.scale_up_threshold:
                scale_up_votes += metric.weight
                reasoning_parts.append(
                    f"{metric.metric_type.value}: {current_value:.2f} > {metric.scale_up_threshold}"
                )
            elif current_value < metric.scale_down_threshold:
                scale_down_votes += metric.weight
                reasoning_parts.append(
                    f"{metric.metric_type.value}: {current_value:.2f} < {metric.scale_down_threshold}"
                )

        # Check predictive scaling
        predicted_load = self.predictive_scaler.predict_load(
            MetricType.REQUEST_RATE, policy.prediction_window
        )

        if predicted_load:
            if predicted_load > 100:  # High predicted load
                scale_up_votes += 0.5
                reasoning_parts.append(f"Predicted high load: {predicted_load:.2f}")

        # Make decision
        action = ScalingAction.NO_ACTION
        target_replicas = current_replicas
        confidence = 0.0

        total_votes = scale_up_votes + scale_down_votes

        if scale_up_votes > scale_down_votes and scale_up_votes > 0:
            if self._can_scale_up(policy):
                action = ScalingAction.SCALE_UP
                target_replicas = min(
                    current_replicas + policy.scale_up_increment,
                    policy.max_replicas
                )
                confidence = min(scale_up_votes / max(total_votes, 1), 1.0)
        elif scale_down_votes > scale_up_votes and scale_down_votes > 0:
            if self._can_scale_down(policy):
                action = ScalingAction.SCALE_DOWN
                target_replicas = max(
                    current_replicas - policy.scale_down_increment,
                    policy.min_replicas
                )
                confidence = min(scale_down_votes / max(total_votes, 1), 1.0)

        reasoning = f"Policy {policy.name}: " + "; ".join(reasoning_parts)

        return ScalingDecision(
            action=action,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            reasoning=reasoning,
            confidence=confidence,
            metrics=metric_scores,
            timestamp=datetime.now()
        )

    def _can_scale_up(self, policy: ScalingPolicy) -> bool:
        """Check if scaling up is allowed."""
        current_time = time.time()
        return (current_time - self.last_scale_time[ScalingAction.SCALE_UP]
                > policy.scale_up_cooldown)

    def _can_scale_down(self, policy: ScalingPolicy) -> bool:
        """Check if scaling down is allowed."""
        current_time = time.time()
        return (current_time - self.last_scale_time[ScalingAction.SCALE_DOWN]
                > policy.scale_down_cooldown)

    def _make_scaling_decision(self, policy_decisions: List[Tuple[ScalingPolicy, ScalingDecision]],
                             current_replicas: int) -> ScalingDecision:
        """Make final scaling decision from multiple policy decisions."""
        if not policy_decisions:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                reasoning="No policy decisions",
                confidence=0.0,
                metrics={},
                timestamp=datetime.now()
            )

        # Weight decisions by confidence
        weighted_scale_up = 0
        weighted_scale_down = 0
        combined_metrics = {}
        reasoning_parts = []

        for policy, decision in policy_decisions:
            if decision.action == ScalingAction.SCALE_UP:
                weighted_scale_up += decision.confidence
            elif decision.action == ScalingAction.SCALE_DOWN:
                weighted_scale_down += decision.confidence

            combined_metrics.update(decision.metrics)
            reasoning_parts.append(f"[{policy.name}] {decision.reasoning}")

        # Final decision
        if weighted_scale_up > weighted_scale_down:
            target_replicas = min(
                max([decision.target_replicas for _, decision in policy_decisions]),
                max([policy.max_replicas for policy, _ in policy_decisions])
            )
            action = ScalingAction.SCALE_UP
            confidence = weighted_scale_up / (weighted_scale_up + weighted_scale_down)
        elif weighted_scale_down > weighted_scale_up:
            target_replicas = max(
                min([decision.target_replicas for _, decision in policy_decisions]),
                max([policy.min_replicas for policy, _ in policy_decisions])
            )
            action = ScalingAction.SCALE_DOWN
            confidence = weighted_scale_down / (weighted_scale_up + weighted_scale_down)
        else:
            action = ScalingAction.NO_ACTION
            target_replicas = current_replicas
            confidence = 0.0

        return ScalingDecision(
            action=action,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            reasoning="; ".join(reasoning_parts),
            confidence=confidence,
            metrics=combined_metrics,
            timestamp=datetime.now()
        )

    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute the scaling decision."""
        logger.info(f"Executing scaling decision: {decision.action.value} "
                   f"{decision.current_replicas} -> {decision.target_replicas} "
                   f"(confidence: {decision.confidence:.2f})")

        success = self.k8s_scaler.scale_deployment(
            self.deployment_name, decision.target_replicas
        )

        if success:
            self.last_scale_time[decision.action] = time.time()
            self.scaling_history.append(decision)

            # Keep only recent history
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]

            logger.info(f"Successfully executed scaling: {decision.reasoning}")
        else:
            logger.error(f"Failed to execute scaling decision: {decision.action.value}")

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        current_replicas = self.k8s_scaler.get_current_replicas(self.deployment_name)

        # Get recent metrics
        recent_metrics = {}
        for metric_type in MetricType:
            value = self.metric_collector.collect_metric(metric_type, self.namespace)
            if value is not None:
                recent_metrics[metric_type.value] = value

        return {
            'deployment_name': self.deployment_name,
            'namespace': self.namespace,
            'current_replicas': current_replicas,
            'policies': {
                name: {
                    'enabled': policy.enabled,
                    'min_replicas': policy.min_replicas,
                    'max_replicas': policy.max_replicas,
                    'metrics_count': len(policy.metrics)
                }
                for name, policy in self.policies.items()
            },
            'recent_metrics': recent_metrics,
            'scaling_history_count': len(self.scaling_history),
            'last_scaling_actions': {
                action.value: self.last_scale_time[action]
                for action in ScalingAction
                if action in self.last_scale_time
            },
            'running': self.running
        }


def create_default_scaling_policies() -> List[ScalingPolicy]:
    """Create default scaling policies for DeepONet."""

    # CPU-based scaling policy
    cpu_policy = ScalingPolicy(
        name="cpu_scaling",
        metrics=[
            ScalingMetric(
                metric_type=MetricType.CPU_UTILIZATION,
                target_value=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                weight=1.0
            )
        ],
        min_replicas=2,
        max_replicas=10,
        scale_up_cooldown=180,
        scale_down_cooldown=300
    )

    # Request-based scaling policy
    request_policy = ScalingPolicy(
        name="request_scaling",
        metrics=[
            ScalingMetric(
                metric_type=MetricType.REQUEST_RATE,
                target_value=100.0,
                scale_up_threshold=150.0,
                scale_down_threshold=50.0,
                weight=1.5
            ),
            ScalingMetric(
                metric_type=MetricType.RESPONSE_TIME,
                target_value=0.5,
                scale_up_threshold=1.0,
                scale_down_threshold=0.2,
                weight=1.2
            )
        ],
        min_replicas=1,
        max_replicas=15,
        scale_up_cooldown=120,
        scale_down_cooldown=240,
        prediction_window=600
    )

    # GPU-based scaling policy (if GPUs are available)
    gpu_policy = ScalingPolicy(
        name="gpu_scaling",
        metrics=[
            ScalingMetric(
                metric_type=MetricType.GPU_UTILIZATION,
                target_value=75.0,
                scale_up_threshold=85.0,
                scale_down_threshold=25.0,
                weight=2.0
            ),
            ScalingMetric(
                metric_type=MetricType.GPU_MEMORY,
                target_value=80.0,
                scale_up_threshold=90.0,
                scale_down_threshold=40.0,
                weight=1.5
            )
        ],
        min_replicas=1,
        max_replicas=5,  # GPUs are expensive
        scale_up_cooldown=300,
        scale_down_cooldown=600,
        enabled=torch.cuda.is_available()
    )

    return [cpu_policy, request_policy, gpu_policy]


if __name__ == "__main__":
    # Example usage
    autoscaler = DeepONetAutoScaler(
        deployment_name="deeponet-api",
        namespace="deeponet",
        prometheus_url="http://localhost:9090"
    )

    # Add default policies
    for policy in create_default_scaling_policies():
        autoscaler.add_scaling_policy(policy)

    # Start auto-scaling
    autoscaler.start_auto_scaling(check_interval=30)

    print("DeepONet Auto-Scaler started")
    print("Monitoring deployment: deeponet-api")
    print("Scaling policies:", list(autoscaler.policies.keys()))

    try:
        # Keep running
        while True:
            status = autoscaler.get_scaling_status()
            print(f"\nCurrent status: {status['current_replicas']} replicas")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping auto-scaler...")
        autoscaler.stop_auto_scaling()