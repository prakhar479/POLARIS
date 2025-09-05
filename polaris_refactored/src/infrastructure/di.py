"""
Dependency Injection Container

Provides a simple but effective dependency injection system for better testability
and flexibility in the POLARIS framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Callable, Optional, Union
import inspect
from functools import wraps

T = TypeVar('T')


class Injectable(ABC):
    """
    Base class for injectable services.
    Services that extend this class can be automatically registered and resolved.
    """
    pass


class DIContainer:
    """
    Dependency Injection Container for POLARIS framework.
    
    Supports:
    - Singleton and transient lifetimes
    - Factory functions
    - Interface to implementation mapping
    - Automatic constructor injection
    """
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._transients: set = set()
    
    def register_singleton(self, interface: Type[T], implementation: Union[Type[T], T]) -> 'DIContainer':
        """Register a service as singleton (one instance for the entire application)."""
        if inspect.isclass(implementation):
            self._services[interface] = implementation
        else:
            # Already instantiated object
            self._singletons[interface] = implementation
        return self
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a service as transient (new instance every time)."""
        self._services[interface] = implementation
        self._transients.add(interface)
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """Register a factory function for creating instances."""
        self._factories[interface] = factory
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        # Check if already instantiated singleton
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check if factory exists
        if interface in self._factories:
            instance = self._factories[interface]()
            if interface not in self._transients:
                self._singletons[interface] = instance
            return instance
        
        # Check if service is registered
        if interface not in self._services:
            raise ValueError(f"Service {interface.__name__} is not registered")
        
        implementation = self._services[interface]
        
        # Create instance with dependency injection
        instance = self._create_instance(implementation)
        
        # Store singleton if not transient
        if interface not in self._transients:
            self._singletons[interface] = instance
        
        return instance
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with automatic dependency injection."""
        # Get constructor signature
        signature = inspect.signature(implementation.__init__)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                # Try to resolve the dependency
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # If dependency can't be resolved and has no default, raise error
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(
                            f"Cannot resolve dependency {param.annotation.__name__} "
                            f"for {implementation.__name__}"
                        )
        
        return implementation(**kwargs)
    
    def clear(self):
        """Clear all registrations (useful for testing)."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._transients.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    return _container


def inject(interface: Type[T]) -> T:
    """Convenience function to resolve a service from the global container."""
    return _container.resolve(interface)


def injectable(cls: Type[T]) -> Type[T]:
    """
    Decorator to mark a class as injectable.
    This is mainly for documentation purposes and future enhancements.
    """
    if not issubclass(cls, Injectable):
        # Add Injectable as a base class if not already present
        cls.__bases__ = (Injectable,) + cls.__bases__
    return cls