"use strict";

const uploadForm = document.querySelector("#upload-form");
if (uploadForm) {
  const input = document.querySelector("#file");
  const dropZone = document.querySelector("#drop-zone");
  const selected = document.querySelector("#selected-file");
  const previewFrame = document.querySelector("#preview-frame");
  const preview = document.querySelector("#file-preview");
  const error = document.querySelector("#upload-error");
  const previewMessage = document.querySelector("#preview-message");
  const button = document.querySelector("#detect-button");
  const isVideo = uploadForm.dataset.kind === "video";
  const ready = uploadForm.dataset.ready === "true";
  const extensions = isVideo ? ["mp4", "mov", "avi", "mkv"] : ["jpg", "jpeg", "png", "webp"];
  let objectUrl = null;
  let processing = false;

  const resetPreview = () => {
    if (isVideo) preview.pause();
    preview.removeAttribute("src");
    if (isVideo) preview.load();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    selected.hidden = true;
    previewFrame.hidden = true;
    previewMessage.hidden = true;
    preview.hidden = false;
  };

  const selectFile = () => {
    resetPreview();
    error.hidden = true;
    button.disabled = true;
    const file = input.files[0];
    if (!file) return;
    const extension = file.name.split(".").pop().toLowerCase();
    let message = "";
    if (!extensions.includes(extension)) message = "文件格式不支持，请选择页面列出的格式。";
    else if (file.size === 0) message = "文件为空，请重新选择。";
    else if (file.size > Number(uploadForm.dataset.limit)) message = "文件超过大小限制，请缩小后上传。";
    if (message) {
      error.textContent = message;
      error.hidden = false;
      input.value = "";
      return;
    }
    document.querySelector("#file-name").textContent = file.name;
    document.querySelector("#file-size").textContent = (file.size / 1024 / 1024).toFixed(2) + " MB";
    selected.hidden = false;
    previewFrame.hidden = false;
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    if (isVideo) preview.load();
    button.disabled = !ready;
  };

  button.disabled = true;
  input.addEventListener("change", selectFile);
  preview.addEventListener("error", () => {
    if (objectUrl) {
      preview.hidden = true;
      previewMessage.hidden = false;
    }
  });
  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!processing) dropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    if (processing) return;
    if (event.dataTransfer.files.length !== 1) {
      error.textContent = "请每次选择一个文件。";
      error.hidden = false;
      return;
    }
    input.files = event.dataTransfer.files;
    selectFile();
  });
  document.querySelector("#remove-file").addEventListener("click", () => {
    if (processing) return;
    input.value = "";
    selectFile();
  });
  const confidenceInput = document.querySelector("#confidence");
  const updateConfidence = () => {
    const value = Number(confidenceInput.value);
    document.querySelector("#confidence-value").value = value.toFixed(2);
    document.querySelector("#threshold-warning").hidden = value < 0.7;
  };
  confidenceInput.addEventListener("input", updateConfidence);
  document.querySelector("#reset-confidence").addEventListener("click", () => {
    if (processing) return;
    confidenceInput.value = "0.25";
    updateConfidence();
  });
  updateConfidence();
  uploadForm.addEventListener("submit", (event) => {
    if (processing || !ready || !input.files.length) {
      event.preventDefault();
      return;
    }
    processing = true;
    button.disabled = true;
    button.textContent = isVideo ? "视频处理中…" : "检测中…";
    document.querySelector("#processing-state").hidden = false;
    document.querySelector("#remove-file").disabled = true;
    document.querySelector("#reset-confidence").disabled = true;
    input.inert = true;
    // 不禁用 file/range 输入，确保浏览器正常发送所选文件与阈值。
    uploadForm.setAttribute("aria-busy", "true");
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) window.location.reload();
  });
}

const deleteForm = document.querySelector(".delete-form");
const deleteDialog = document.querySelector("#delete-dialog");
if (deleteForm && deleteDialog) {
  deleteForm.addEventListener("submit", (event) => {
    event.preventDefault();
    deleteDialog.showModal();
  });
  document.querySelector("#cancel-delete").addEventListener("click", () => deleteDialog.close());
  document.querySelector("#confirm-delete").addEventListener("click", () => {
    document.querySelector("#confirm-delete").disabled = true;
    deleteForm.submit();
  });
}
