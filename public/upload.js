// Handle the form submission for file upload
document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the default form submission

    const fileInput = document.getElementById('fileUpload'); // Updated to match HTML ID
    if (!fileInput.files[0]) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('dataset', fileInput.files[0]);

    // Send the file to the server using POST
    fetch('http://localhost:3000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => alert('File uploaded successfully: ' + data))
    .catch(error => {
        console.error('Error uploading file:', error);
        alert('Error uploading file. Check console for details.');
    });
    
});

