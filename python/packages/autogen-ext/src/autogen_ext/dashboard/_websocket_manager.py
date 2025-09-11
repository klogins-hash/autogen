"""
WebSocket connection management for real-time dashboard updates.
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class ClientConnection:
    """Represents a WebSocket client connection."""
    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketManager:
    """Manages WebSocket connections for the dashboard."""
    
    def __init__(self):
        # Client connections
        self.connections: Dict[str, ClientConnection] = {}
        self.agent_connections: Dict[str, ClientConnection] = {}
        
        # Subscription management
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Message queues for offline agents
        self.agent_message_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Connection statistics
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0
        }
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Connect a new WebSocket client."""
        
        await websocket.accept()
        
        if not client_id:
            client_id = f"client_{int(time.time() * 1000)}"
        
        connection = ClientConnection(
            websocket=websocket,
            client_id=client_id
        )
        
        self.connections[client_id] = connection
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] += 1
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": time.time()
        })
        
        return client_id
    
    async def connect_agent(self, websocket: WebSocket, agent_id: str) -> None:
        """Connect an agent WebSocket."""
        
        await websocket.accept()
        
        connection = ClientConnection(
            websocket=websocket,
            client_id=agent_id,
            agent_id=agent_id
        )
        
        self.agent_connections[agent_id] = connection
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] += 1
        
        # Send queued messages to agent
        if agent_id in self.agent_message_queues:
            for message in self.agent_message_queues[agent_id]:
                await self.send_to_agent(agent_id, message)
            
            # Clear the queue
            self.agent_message_queues[agent_id].clear()
        
        # Notify dashboard clients about agent connection
        await self.broadcast({
            "type": "agent_connected",
            "agent_id": agent_id,
            "timestamp": time.time()
        }, topic="agent_status")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client."""
        
        # Find and remove the connection
        client_id = None
        for cid, connection in self.connections.items():
            if connection.websocket == websocket:
                client_id = cid
                break
        
        if client_id:
            connection = self.connections[client_id]
            
            # Remove from subscriptions
            for topic in connection.subscriptions:
                self.topic_subscribers[topic].discard(client_id)
            
            # Remove connection
            del self.connections[client_id]
            self.connection_stats["active_connections"] -= 1
    
    async def disconnect_agent(self, agent_id: str) -> None:
        """Disconnect an agent WebSocket."""
        
        if agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
            self.connection_stats["active_connections"] -= 1
            
            # Notify dashboard clients about agent disconnection
            await self.broadcast({
                "type": "agent_disconnected",
                "agent_id": agent_id,
                "timestamp": time.time()
            }, topic="agent_status")
    
    async def disconnect_all(self) -> None:
        """Disconnect all WebSocket connections."""
        
        # Disconnect regular clients
        for connection in list(self.connections.values()):
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        # Disconnect agents
        for connection in list(self.agent_connections.values()):
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        self.connections.clear()
        self.agent_connections.clear()
        self.topic_subscribers.clear()
        self.connection_stats["active_connections"] = 0
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific client."""
        
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            self.connection_stats["messages_sent"] += 1
            return True
        
        except WebSocketDisconnect:
            await self.disconnect(connection.websocket)
            return False
        
        except Exception as e:
            print(f"Error sending message to client {client_id}: {e}")
            return False
    
    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific agent."""
        
        if agent_id in self.agent_connections:
            connection = self.agent_connections[agent_id]
            
            try:
                await connection.websocket.send_text(json.dumps(message))
                self.connection_stats["messages_sent"] += 1
                return True
            
            except WebSocketDisconnect:
                await self.disconnect_agent(agent_id)
                return False
            
            except Exception as e:
                print(f"Error sending message to agent {agent_id}: {e}")
                return False
        
        else:
            # Queue message for when agent connects
            self.agent_message_queues[agent_id].append(message)
            return True
    
    async def broadcast(self, message: Dict[str, Any], topic: Optional[str] = None) -> int:
        """Broadcast message to all clients or subscribers of a topic."""
        
        sent_count = 0
        
        if topic:
            # Send to topic subscribers
            subscriber_ids = self.topic_subscribers.get(topic, set())
            
            for client_id in list(subscriber_ids):
                if await self.send_to_client(client_id, message):
                    sent_count += 1
        
        else:
            # Send to all clients
            for client_id in list(self.connections.keys()):
                if await self.send_to_client(client_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_agents(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected agents."""
        
        sent_count = 0
        
        for agent_id in list(self.agent_connections.keys()):
            if await self.send_to_agent(agent_id, message):
                sent_count += 1
        
        return sent_count
    
    async def add_subscription(self, websocket: WebSocket, topics: List[str]) -> bool:
        """Add client subscription to topics."""
        
        # Find client
        client_id = None
        for cid, connection in self.connections.items():
            if connection.websocket == websocket:
                client_id = cid
                break
        
        if not client_id:
            return False
        
        connection = self.connections[client_id]
        
        for topic in topics:
            connection.subscriptions.add(topic)
            self.topic_subscribers[topic].add(client_id)
        
        return True
    
    async def remove_subscription(self, websocket: WebSocket, topics: List[str]) -> bool:
        """Remove client subscription from topics."""
        
        # Find client
        client_id = None
        for cid, connection in self.connections.items():
            if connection.websocket == websocket:
                client_id = cid
                break
        
        if not client_id:
            return False
        
        connection = self.connections[client_id]
        
        for topic in topics:
            connection.subscriptions.discard(topic)
            self.topic_subscribers[topic].discard(client_id)
        
        return True
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        
        return {
            "stats": self.connection_stats.copy(),
            "active_clients": len(self.connections),
            "active_agents": len(self.agent_connections),
            "topics": {
                topic: len(subscribers)
                for topic, subscribers in self.topic_subscribers.items()
            },
            "queued_messages": {
                agent_id: len(messages)
                for agent_id, messages in self.agent_message_queues.items()
                if messages
            }
        }
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        
        if client_id in self.connections:
            connection = self.connections[client_id]
            return {
                "client_id": client_id,
                "connected_at": connection.connected_at,
                "subscriptions": list(connection.subscriptions),
                "metadata": connection.metadata
            }
        
        return None
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        
        if agent_id in self.agent_connections:
            connection = self.agent_connections[agent_id]
            return {
                "agent_id": agent_id,
                "connected_at": connection.connected_at,
                "queued_messages": len(self.agent_message_queues.get(agent_id, [])),
                "metadata": connection.metadata
            }
        
        return None
    
    async def ping_all_connections(self) -> Dict[str, bool]:
        """Ping all connections to check if they're alive."""
        
        results = {}
        
        # Ping regular clients
        for client_id, connection in list(self.connections.items()):
            try:
                await connection.websocket.ping()
                results[f"client_{client_id}"] = True
            except Exception:
                results[f"client_{client_id}"] = False
                await self.disconnect(connection.websocket)
        
        # Ping agents
        for agent_id, connection in list(self.agent_connections.items()):
            try:
                await connection.websocket.ping()
                results[f"agent_{agent_id}"] = True
            except Exception:
                results[f"agent_{agent_id}"] = False
                await self.disconnect_agent(agent_id)
        
        return results
    
    async def send_heartbeat(self) -> None:
        """Send heartbeat to all connections."""
        
        heartbeat_message = {
            "type": "heartbeat",
            "timestamp": time.time()
        }
        
        await self.broadcast(heartbeat_message)
        await self.broadcast_to_agents(heartbeat_message)
    
    def update_client_metadata(self, client_id: str, metadata: Dict[str, Any]) -> bool:
        """Update client metadata."""
        
        if client_id in self.connections:
            self.connections[client_id].metadata.update(metadata)
            return True
        
        return False
    
    def update_agent_metadata(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Update agent metadata."""
        
        if agent_id in self.agent_connections:
            self.agent_connections[agent_id].metadata.update(metadata)
            return True
        
        return False
    
    async def cleanup_stale_connections(self, timeout_seconds: int = 300) -> int:
        """Clean up stale connections that haven't been active."""
        
        current_time = time.time()
        cleaned_count = 0
        
        # Check regular clients
        for client_id, connection in list(self.connections.items()):
            if current_time - connection.connected_at > timeout_seconds:
                try:
                    await connection.websocket.close()
                    await self.disconnect(connection.websocket)
                    cleaned_count += 1
                except Exception:
                    pass
        
        # Check agents
        for agent_id, connection in list(self.agent_connections.items()):
            if current_time - connection.connected_at > timeout_seconds:
                try:
                    await connection.websocket.close()
                    await self.disconnect_agent(agent_id)
                    cleaned_count += 1
                except Exception:
                    pass
        
        return cleaned_count
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get list of subscribers for a topic."""
        
        return list(self.topic_subscribers.get(topic, set()))
    
    def get_client_subscriptions(self, client_id: str) -> List[str]:
        """Get list of topics a client is subscribed to."""
        
        if client_id in self.connections:
            return list(self.connections[client_id].subscriptions)
        
        return []
    
    async def notify_topic_update(self, topic: str, data: Any) -> int:
        """Notify subscribers about a topic update."""
        
        message = {
            "type": "topic_update",
            "topic": topic,
            "data": data,
            "timestamp": time.time()
        }
        
        return await self.broadcast(message, topic=topic)
