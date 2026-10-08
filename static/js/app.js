// IntelliSum Phase 1 — client-side validation + loading indicator.
(function () {
  var form = document.getElementById("upload-form");
  if (!form) return;

  var fileInput = document.getElementById("file");
  var errBox = document.getElementById("form-error");
  var loading = document.getElementById("loading");
  var submitBtn = document.getElementById("submit-btn");
  var MAX_BYTES = 16 * 1024 * 1024;
  var ALLOWED = ["pdf", "docx", "txt"];

  function showError(msg) {
    errBox.textContent = msg;
    errBox.style.display = "block";
  }
  function clearError() {
    errBox.textContent = "";
    errBox.style.display = "none";
  }

  form.addEventListener("submit", function (e) {
    clearError();
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
    if (loading) loading.style.display = "block";
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Processing…";
    }
  });
})();
