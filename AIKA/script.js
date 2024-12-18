window.onload = function() {
    const splashScreen = document.getElementById("splash-screen");
    const content = document.querySelector(".content");

    // Wait 5 seconds before starting the slide-up animation
    setTimeout(function() {
        // Start the slide-up animation after 5 seconds
        splashScreen.style.animation = "slideUp 1s forwards"; // Apply the slideUp animation

        // After the animation is done, hide splash screen and show content
        setTimeout(function() {
            splashScreen.style.display = "none";  // Hide splash screen
            content.style.display = "block";  // Show the main content
        }, 1000);  // 1 second (time matching the slideUp animation duration)
    }, 1000);  // 5 seconds delay before starting the animation
};


const clickableDiv = document.querySelector('.papers');
const slidingDiv = document.querySelector('.sliding-div');
const slidContents = document.querySelectorAll('.slid-content');

// Toggle the main sliding-div
clickableDiv.addEventListener('click', () => {
    slidingDiv.classList.toggle('hidden');
    slidingDiv.classList.toggle('visible');
});

// Add functionality to each slid-content
slidContents.forEach((content) => {
    content.addEventListener('click', (e) => {
        // Prevent bubbling up to parent divs
        e.stopPropagation();

        // Toggle open/close for this slid-content
        content.classList.toggle('open');
    });
});


// Select the toggle and text elements
const toggle = document.getElementById('toggle-dark-mode');

// Add event listener for toggle
toggle.addEventListener('change', () => {
  if (toggle.checked) {
    document.body.classList.remove('light-mode');
  } else {
    document.body.classList.add('light-mode');
  }
});
