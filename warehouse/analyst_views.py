"""
Spreadsheet analyst views.

The chat endpoint is synchronous: a turn can take 10-60s while the model
thinks and pandas runs. That is fine behind gunicorn with a raised
`--timeout`, and it keeps the code readable. If your traffic makes it a
problem, move `submit_question` into a Celery task and have `analyst_ask`
return a task id the page polls - the service function is already written to
be called from anywhere.
"""

from __future__ import annotations

import json

from django.contrib import messages as django_messages
from django.contrib.auth.decorators import login_required
from django.db.models import Prefetch
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.views.decorators.http import require_GET, require_POST

from .forms import AnalystQuestionForm, GoogleSheetLinkForm, WorkbookUploadForm
from .models import Artifact, ChatMessage, Conversation, Workbook
from .services.analyst import drop_session, profile_workbook, submit_question
from .services.google_sheets_import import GoogleSheetImportError, fetch_google_sheet_as_xlsx


def _owned(model, request: HttpRequest, **kwargs):
    """Fetch scoped to the requesting user.

    Drop the owner filter if you are wiring this into a page where any
    authenticated user may see any workbook.
    """
    obj = get_object_or_404(model, **kwargs)
    if obj.owner_id and obj.owner_id != request.user.id:
        raise Http404
    return obj


@login_required
def analyst_workbook_list(request: HttpRequest) -> HttpResponse:
    form = WorkbookUploadForm()
    sheet_form = GoogleSheetLinkForm()
    workbooks = Workbook.objects.filter(owner=request.user)
    return render(
        request,
        "warehouse/analyst/workbook_list.html",
        {
            "form": form,
            "sheet_form": sheet_form,
            "workbooks": workbooks,
            "sidebar_section": "warehouse",
        },
    )


@login_required
@require_POST
def analyst_workbook_upload(request: HttpRequest) -> HttpResponse:
    form = WorkbookUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        for error in form.errors.get("file", []):
            django_messages.error(request, error)
        return redirect("warehouse:analyst_workbook_list")

    workbook: Workbook = form.save(commit=False)
    workbook.owner = request.user
    workbook.original_name = request.FILES["file"].name
    workbook.save()

    profile_workbook(workbook)
    if workbook.load_error:
        django_messages.error(request, f"Could not read that file. {workbook.load_error}")
        return redirect("warehouse:analyst_workbook_list")

    conversation = Conversation.objects.create(workbook=workbook, owner=request.user)
    return redirect(conversation.get_absolute_url())


@login_required
@require_POST
def analyst_link_google_sheet(request: HttpRequest) -> HttpResponse:
    """Link a Google Sheet in place of an upload — fetched once via the
    Sheets API and saved as a real .xlsx, so everything downstream
    (profiling, chat, sandboxed pandas) runs through the exact same path
    as an uploaded file."""
    from django.core.files.base import ContentFile

    form = GoogleSheetLinkForm(request.POST)
    if not form.is_valid():
        for error in form.errors.get("id_or_url", []):
            django_messages.error(request, error)
        return redirect("warehouse:analyst_workbook_list")

    spreadsheet_id = form.cleaned_data["id_or_url"]
    try:
        xlsx_bytes, title = fetch_google_sheet_as_xlsx(spreadsheet_id)
    except GoogleSheetImportError as exc:
        django_messages.error(request, str(exc))
        return redirect("warehouse:analyst_workbook_list")

    workbook = Workbook(
        owner=request.user,
        original_name=f"{title}.xlsx",
        source_type=Workbook.SOURCE_GOOGLE_SHEET,
        google_sheet_id=spreadsheet_id,
        google_sheet_url=f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
    )
    workbook.file.save(f"{title}.xlsx", ContentFile(xlsx_bytes), save=False)
    workbook.save()

    profile_workbook(workbook)
    if workbook.load_error:
        django_messages.error(request, f"Could not read that spreadsheet. {workbook.load_error}")
        return redirect("warehouse:analyst_workbook_list")

    conversation = Conversation.objects.create(workbook=workbook, owner=request.user)
    return redirect(conversation.get_absolute_url())


@login_required
@require_POST
def analyst_refresh_google_sheet(request: HttpRequest, pk) -> HttpResponse:
    """Re-fetch a linked Google Sheet's current data and re-profile it."""
    from django.core.files.base import ContentFile

    workbook = _owned(Workbook, request, pk=pk)
    if workbook.source_type != Workbook.SOURCE_GOOGLE_SHEET or not workbook.google_sheet_id:
        raise Http404

    try:
        xlsx_bytes, title = fetch_google_sheet_as_xlsx(workbook.google_sheet_id)
    except GoogleSheetImportError as exc:
        django_messages.error(request, str(exc))
        return redirect(workbook.get_absolute_url())

    workbook.file.delete(save=False)
    workbook.file.save(f"{title}.xlsx", ContentFile(xlsx_bytes), save=False)
    workbook.save()
    profile_workbook(workbook)
    if workbook.load_error:
        django_messages.error(request, f"Could not read that spreadsheet. {workbook.load_error}")
    else:
        django_messages.success(request, "Refreshed from Google Sheets.")
    return redirect(workbook.get_absolute_url())


@login_required
def analyst_workbook_detail(request: HttpRequest, pk) -> HttpResponse:
    workbook = _owned(Workbook, request, pk=pk)
    return render(
        request,
        "warehouse/analyst/workbook_detail.html",
        {"workbook": workbook, "sidebar_section": "warehouse"},
    )


@login_required
@require_POST
def analyst_new_conversation(request: HttpRequest, pk) -> HttpResponse:
    """Start a fresh conversation against an already-uploaded workbook."""
    workbook = _owned(Workbook, request, pk=pk)
    conversation = Conversation.objects.create(workbook=workbook, owner=request.user)
    return redirect(conversation.get_absolute_url())


@login_required
def analyst_chat(request: HttpRequest, pk) -> HttpResponse:
    conversation = _owned(
        Conversation.objects.select_related("workbook").prefetch_related(
            Prefetch("messages", queryset=ChatMessage.objects.prefetch_related("artifacts"))
        ),
        request,
        pk=pk,
    )
    return render(
        request,
        "warehouse/analyst/chat.html",
        {
            "conversation": conversation,
            "workbook": conversation.workbook,
            "form": AnalystQuestionForm(),
            "sidebar_section": "warehouse",
            "suggestions": [
                "Profile this workbook and tell me what's in it",
                "What are the main data quality problems?",
                "Show the key trend as a chart",
                "Write a full analysis report",
            ],
        },
    )


@login_required
@require_POST
def analyst_ask(request: HttpRequest, pk) -> JsonResponse:
    """Run one turn and return the rendered assistant message as JSON."""
    conversation = _owned(Conversation, request, pk=pk)

    if request.content_type == "application/json":
        payload = json.loads(request.body or "{}")
        form = AnalystQuestionForm(payload)
    else:
        form = AnalystQuestionForm(request.POST)

    if not form.is_valid():
        return JsonResponse({"ok": False, "error": "Please enter a question."}, status=400)

    message = submit_question(conversation, form.cleaned_data["question"])

    # Render server-side: model output is untrusted, and this keeps exactly one
    # sanitisation path shared by the full page and the incremental update.
    html = render_to_string(
        "warehouse/analyst/_message.html", {"message": message}, request=request
    )

    return JsonResponse(
        {
            "ok": message.role != "error",
            "role": message.role,
            "content": message.content,
            "html": html,
            "artifacts": [
                {"kind": a.kind, "title": a.title, "url": a.file.url}
                for a in message.artifacts.all()
            ],
        }
    )


@login_required
@require_POST
def analyst_reset_kernel(request: HttpRequest, pk) -> JsonResponse:
    """Drop the cached pandas kernel - the next turn reloads the workbook."""
    conversation = _owned(Conversation, request, pk=pk)
    drop_session(str(conversation.id))
    return JsonResponse({"ok": True})


@login_required
@require_GET
def analyst_artifact_download(request: HttpRequest, pk) -> HttpResponse:
    artifact = get_object_or_404(Artifact.objects.select_related("conversation"), pk=pk)
    owner_id = artifact.conversation.owner_id
    if owner_id and owner_id != request.user.id:
        raise Http404
    response = HttpResponse(
        artifact.file.read(), content_type="application/octet-stream"
    )
    response["Content-Disposition"] = f'attachment; filename="{artifact.file.name.split("/")[-1]}"'
    return response
