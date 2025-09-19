"""
Distributed Tracing System for POLARIS

Provides end-to-end tracing capabilities with automatic instrumentation,
trace correlation across component boundaries, and integration with observability systems.
"""

import functools
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import defaultdict


# Context variables for trace propagation
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar('parent_span_id', default=None)

F = TypeVar('F', bound=Callable[..., Any])


class SpanKind(Enum):
    """Types of spans in distributed tracing"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanEvent:
    """Event within a span"""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Represents a single span in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    
    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish the span"""
        self.end_time = datetime.utcnow()
        self.status = status
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span"""
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def set_error(self, error: Exception) -> None:
        """Mark span as error and add error details"""
        self.status = SpanStatus.ERROR
        self.add_tag("error", True)
        self.add_tag("error.type", type(error).__name__)
        self.add_tag("error.message", str(error))
        self.add_event("exception", {
            "exception.type": type(error).__name__,
            "exception.message": str(error),
            "exception.module": type(error).__module__
        })


@dataclass
class TraceContext:
    """Context for trace propagation"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert trace context to HTTP headers"""
        headers = {
            'X-Trace-Id': self.trace_id,
            'X-Span-Id': self.span_id
        }
        if self.parent_span_id:
            headers['X-Parent-Span-Id'] = self.parent_span_id
        
        # Add baggage
        for key, value in self.baggage.items():
            headers[f'X-Baggage-{key}'] = value
        
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create trace context from HTTP headers"""
        trace_id = headers.get('X-Trace-Id')
        span_id = headers.get('X-Span-Id')
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get('X-Parent-Span-Id')
        
        # Extract baggage
        baggage = {}
        for key, value in headers.items():
            if key.startswith('X-Baggage-'):
                baggage_key = key[10:]  # Remove 'X-Baggage-' prefix
                baggage[baggage_key] = value
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )


class SpanExporter(ABC):
    """Abstract base class for span exporters"""
    
    @abstractmethod
    def export(self, spans: List[Span]) -> None:
        """Export spans to external system"""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Console exporter for development/debugging"""
    
    def export(self, spans: List[Span]) -> None:
        """Print spans to console"""
        for span in spans:
            print(f"Span: {span.operation_name}")
            print(f"  Trace ID: {span.trace_id}")
            print(f"  Span ID: {span.span_id}")
            print(f"  Parent: {span.parent_span_id}")
            print(f"  Duration: {span.duration_ms}ms")
            print(f"  Status: {span.status.value}")
            print(f"  Tags: {span.tags}")
            if span.events:
                print(f"  Events: {len(span.events)}")
            print()


class JaegerSpanExporter(SpanExporter):
    """Jaeger-compatible span exporter"""
    
    def __init__(self, endpoint: str, service_name: str = "polaris"):
        self.endpoint = endpoint
        self.service_name = service_name
    
    def export(self, spans: List[Span]) -> None:
        """Export spans to Jaeger (placeholder implementation)"""
        # In a real implementation, this would send spans to Jaeger
        # For now, we'll just log the export
        print(f"Exporting {len(spans)} spans to Jaeger at {self.endpoint}")


class PolarisTracer:
    """
    Distributed tracer for POLARIS system.
    
    Provides end-to-end tracing capabilities with automatic instrumentation,
    trace correlation across component boundaries, and integration with
    external tracing systems like Jaeger or Zipkin.
    """
    
    def __init__(self, service_name: str = "polaris"):
        self.service_name = service_name
        self.exporters: List[SpanExporter] = []
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: List[Span] = []
        self.max_completed_spans = 1000  # Prevent memory leaks
    
    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add a span exporter"""
        self.exporters.append(exporter)
    
    def start_span(
        self,
        operation_name: str,
        parent_context: Optional[TraceContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        
        # Generate IDs
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            # Check if we're in an existing trace context
            current_trace_id = trace_id_var.get()
            current_span_id = span_id_var.get()
            
            if current_trace_id and current_span_id:
                trace_id = current_trace_id
                parent_span_id = current_span_id
            else:
                trace_id = str(uuid.uuid4())
                parent_span_id = None
        
        span_id = str(uuid.uuid4())
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            kind=kind
        )
        
        # Add default tags
        span.add_tag("service.name", self.service_name)
        span.add_tag("span.kind", kind.value)
        
        # Add custom tags
        if tags:
            for key, value in tags.items():
                span.add_tag(key, value)
        
        # Store active span
        self.active_spans[span_id] = span
        
        return span
    
    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish a span"""
        span.finish(status)
        
        # Move from active to completed
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        self.completed_spans.append(span)
        
        # Prevent memory leaks
        if len(self.completed_spans) > self.max_completed_spans:
            self._export_and_clear_spans()
    
    def _export_and_clear_spans(self) -> None:
        """Export completed spans and clear the buffer"""
        if self.completed_spans:
            for exporter in self.exporters:
                try:
                    exporter.export(self.completed_spans.copy())
                except Exception as e:
                    print(f"Failed to export spans: {e}")
            
            self.completed_spans.clear()
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing an operation"""
        span = self.start_span(operation_name, kind=kind, tags=tags)
        
        # Set context variables
        trace_token = trace_id_var.set(span.trace_id)
        span_token = span_id_var.set(span.span_id)
        parent_token = parent_span_id_var.set(span.parent_span_id)
        
        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
        finally:
            # Reset context variables
            trace_id_var.reset(trace_token)
            span_id_var.reset(span_token)
            parent_span_id_var.reset(parent_token)
    
    @contextmanager
    def trace_adaptation_flow(self, adaptation_id: str, system_id: str):
        """Context manager for tracing complete adaptation flow"""
        tags = {
            "adaptation.id": adaptation_id,
            "system.id": system_id,
            "component": "adaptation_flow"
        }
        
        with self.trace_operation("adaptation_flow", SpanKind.INTERNAL, tags) as span:
            span.add_event("adaptation_started", {"adaptation_id": adaptation_id})
            yield span
            span.add_event("adaptation_completed", {"adaptation_id": adaptation_id})
    
    @contextmanager
    def trace_telemetry_processing(self, system_id: str, event_type: str):
        """Context manager for tracing telemetry processing"""
        tags = {
            "system.id": system_id,
            "telemetry.event_type": event_type,
            "component": "telemetry_processing"
        }
        
        with self.trace_operation("telemetry_processing", SpanKind.CONSUMER, tags) as span:
            yield span
    
    @contextmanager
    def trace_world_model_update(self, system_id: str, model_type: str):
        """Context manager for tracing world model updates"""
        tags = {
            "system.id": system_id,
            "world_model.type": model_type,
            "component": "world_model"
        }
        
        with self.trace_operation("world_model_update", SpanKind.INTERNAL, tags) as span:
            yield span
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context"""
        trace_id = trace_id_var.get()
        span_id = span_id_var.get()
        parent_span_id = parent_span_id_var.get()
        
        if trace_id and span_id:
            return TraceContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id
            )
        return None
    
    def inject_context(self, context: TraceContext) -> None:
        """Inject trace context into current execution"""
        trace_id_var.set(context.trace_id)
        span_id_var.set(context.span_id)
        parent_span_id_var.set(context.parent_span_id)
    
    def flush(self) -> None:
        """Flush all pending spans"""
        self._export_and_clear_spans()


def trace_polaris_method(
    operation_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tags: Optional[Dict[str, Any]] = None
):
    """
    Decorator for automatic method tracing.
    
    Usage:
        @trace_polaris_method("my_operation")
        async def my_async_method(self):
            pass
        
        @trace_polaris_method(tags={"component": "adapter"})
        def my_sync_method(self):
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            
            with tracer.trace_operation(op_name, kind, tags):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
            
            with tracer.trace_operation(op_name, kind, tags):
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global tracer instance
_tracer: Optional[PolarisTracer] = None


def get_tracer() -> PolarisTracer:
    """Get the global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = PolarisTracer()
    return _tracer


def configure_tracing(
    service_name: str = "polaris",
    console_export: bool = True,
    jaeger_endpoint: Optional[str] = None
) -> None:
    """Configure distributed tracing for POLARIS"""
    global _tracer
    _tracer = PolarisTracer(service_name)
    
    if console_export:
        _tracer.add_exporter(ConsoleSpanExporter())
    
    if jaeger_endpoint:
        _tracer.add_exporter(JaegerSpanExporter(jaeger_endpoint, service_name))


# Convenience functions for getting current trace information
def get_trace_id() -> Optional[str]:
    """Get current trace ID"""
    return trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Get current span ID"""
    return span_id_var.get()


def get_parent_span_id() -> Optional[str]:
    """Get current parent span ID"""
    return parent_span_id_var.get()