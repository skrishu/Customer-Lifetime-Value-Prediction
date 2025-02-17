// Function to show the selected section and hide the others
    function showSection(sectionId) {
        // Hide all sections
        const sections = document.querySelectorAll('.section');
        sections.forEach(section => {
            section.classList.remove('active');
        });

        // Show the clicked section
        const activeSection = document.getElementById(sectionId);
        activeSection.classList.add('active');
    }

    // Default to show the 'home' section when the page loads
    document.addEventListener('DOMContentLoaded', () => {
        showSection('home');
    });

    // Add this to your script.js file
    function uploadFile() {
        const fileInput = document.getElementById('fileUpload');
        const file = fileInput.files[0];
        
        if (!file) {
            alert("Please choose a file to upload.");
            return;
        }
      
        const formData = new FormData();
        formData.append('dataset', file);
      
        // Send the file to the server using POST
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => alert(data))  // Check if upload was successful
        .catch(error => console.error('Error:', error));
      }
      
