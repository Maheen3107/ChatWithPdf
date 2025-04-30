// Handle PDF upload and processing
document.getElementById('submit-btn').addEventListener('click', function() {
    const pdfFiles = document.getElementById('pdf-upload').files;
    const formData = new FormData();
    for (let i = 0; i < pdfFiles.length; i++) {
        formData.append('pdf_files', pdfFiles[i]);
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('upload-message').innerText = data.message;
    })
    .catch(error => console.error('Error:', error));
});

// Handle user question submission
document.getElementById('ask-btn').addEventListener('click', function() {
    const question = document.getElementById('user-question').value;
    const textChunks = []; // You should replace this with actual chunks from the backend
    
    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, text_chunks: textChunks })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('response-text').innerText = 'Answer: ' + data.answer;
    })
    .catch(error => console.error('Error:', error));
});
