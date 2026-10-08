// IntelliSum — client-side validation + loading indicator.
// Local extractive NLP only (no AI method selection).
(function () {
  function wire(formId, btnId, needsFile) {
    var form = document.getElementById(formId);
    if (!form) return;

    var fileInput = document.getElementById("file");
    var errBox = document.getElementById("form-error");
    var loading = document.getElementById("loading");
    var submitBtn = document.getElementById(btnId);
    var MAX_BYTES = 16 * 1024 * 1024;
    var ALLOWED = ["pdf", "docx", "txt"];

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
          showError("File too large. Maximum allowed size is 16 MB.");
          e.preventDefault();
          return;
        }
      }
      if (loading) {
        loading.style.display = "block";
        loading.textContent = "Processing document…";
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
