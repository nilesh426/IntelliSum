// IntelliSum — client-side validation, file info + staged loading indicator.
// Local extractive NLP only (no AI method selection).
(function () {
  var STAGES = [
    "Analyzing document…",
    "Extracting text…",
    "Analyzing sentences…",
    "Calculating sentence importance…",
    "Preparing structured summary…"
  ];

  function formatSize(bytes) {
    if (!bytes && bytes !== 0) return "";
    if (bytes < 1024) return bytes + " bytes";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(2) + " MB";
  }

  function wire(formId, btnId, needsFile) {
    var form = document.getElementById(formId);
    if (!form) return;

    var fileInput = document.getElementById("file");
    var errBox = document.getElementById("form-error");
    var loading = document.getElementById("loading");
    var submitBtn = document.getElementById(btnId);
    var fileInfo = document.getElementById("file-info");
    var maxAttr = form.getAttribute("data-max-bytes");
    var MAX_BYTES = maxAttr ? parseInt(maxAttr, 10) : 16 * 1024 * 1024;
    var MAX_MB = Math.round(MAX_BYTES / (1024 * 1024));
    var ALLOWED = ["pdf", "docx", "txt"];
    var stageTimer = null;

    function showError(msg) {
      if (!errBox) return;
      errBox.textContent = msg;
      errBox.style.display = "block";
    }
    function clearError() {
      if (!errBox) return;
      errBox.textContent = "";
      errBox.style.display = "none";
    }

    function renderFileInfo() {
      if (!fileInfo || !fileInput || !fileInput.files || !fileInput.files.length) {
        if (fileInfo) fileInfo.style.display = "none";
        return;
      }
      var f = fileInput.files[0];
      var ext = (f.name.split(".").pop() || "").toLowerCase();
      fileInfo.textContent =
        "Selected: " + f.name + " (" + ext.toUpperCase() + ", " + formatSize(f.size) + ")";
      fileInfo.style.display = "block";
    }

    if (fileInput && needsFile) {
      fileInput.addEventListener("change", function () {
        clearError();
        renderFileInfo();
        var files = fileInput.files;
        if (!files || !files.length) return;
        var f = files[0];
        var ext = (f.name.split(".").pop() || "").toLowerCase();
        if (ALLOWED.indexOf(ext) === -1) {
          showError("Unsupported format. Please upload PDF, DOCX or TXT.");
        } else if (f.size === 0) {
          showError("The selected file is empty.");
        } else if (f.size > MAX_BYTES) {
          showError("File too large. Maximum allowed size is " + MAX_MB + " MB.");
        }
      });
    }

    form.addEventListener("submit", function (e) {
      clearError();
      if (needsFile) {
        var files = fileInput.files;
        if (!files || files.length === 0) {
          showError("Please choose a file first.");
          e.preventDefault();
          return;
        }
        var f = files[0];
        var ext = (f.name.split(".").pop() || "").toLowerCase();
        if (ALLOWED.indexOf(ext) === -1) {
          showError("Unsupported format. Please upload PDF, DOCX or TXT.");
          e.preventDefault();
          return;
        }
        if (f.size === 0) {
          showError("The selected file is empty.");
          e.preventDefault();
          return;
        }
        if (f.size > MAX_BYTES) {
          showError("File too large. Maximum allowed size is " + MAX_MB + " MB.");
          e.preventDefault();
          return;
        }
      }
      if (loading) {
        // Staged progress: purely client-side, traditional-NLP wording.
        var stage = 0;
        loading.style.display = "block";
        loading.textContent = "Generating NLP summary… " + STAGES[0];
        stageTimer = setInterval(function () {
          stage = (stage + 1) % STAGES.length;
          loading.textContent = "Generating NLP summary… " + STAGES[stage];
        }, 1400);
        window.addEventListener("pagehide", function () {
          if (stageTimer) clearInterval(stageTimer);
        });
      }
      if (submitBtn) {
        submitBtn.disabled = true; // prevent accidental duplicate submissions
        submitBtn.textContent = "Processing…";
      }
    });
  }

  wire("upload-form", "submit-btn", true);
  wire("resummarize-form", "resubmit-btn", false);
})();
