"""DRF API views for the agents layer."""
from __future__ import annotations

import logging

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

logger = logging.getLogger("agents")


class RunAgentsView(APIView):
    """Invoke the multi-agent graph with a task.

    POST /api/v1/agents/run/

    Request body:
        task     (str, required)  — the task or question for the agents
        facility (str, optional)  — facility name for data-scoping

    Response:
        final_response  — synthesised answer from all agents
        agent_outputs   — raw output keyed by agent name
        evaluation      — supervisor's quality assessment
        iterations      — number of supervisor routing cycles used
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        task = (request.data.get("task") or "").strip()
        if not task:
            return Response(
                {"error": "The 'task' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        facility = (request.data.get("facility") or "").strip() or None

        from authentication.roles import get_user_role
        user_role = get_user_role(request.user)

        from agents.graph import run_agents
        result = run_agents(task=task, user_role=user_role, facility=facility)

        logger.info(
            "Agent run completed | user=%s | role=%s | iterations=%d",
            request.user.username,
            user_role,
            result.get("iterations", 0),
        )

        return Response(result, status=status.HTTP_200_OK)
