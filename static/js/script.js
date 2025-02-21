document.addEventListener("DOMContentLoaded", function () { 
    console.log("JavaScript Loaded");

    // Form validation for symptoms search
    let form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function (event) {
            let symptomsInput = document.querySelector("input[name='symptoms']");
            let ageInput = document.querySelector("input[name='age']");
            let genderInput = document.querySelector("select[name='Gender']");

            console.log("Debugging Form Input:");
            console.log("Symptoms:", symptomsInput ? symptomsInput.value.trim() : "Not Found");
            console.log("Age:", ageInput ? ageInput.value.trim() : "Not Found");
            console.log("Gender:", genderInput ? genderInput.value : "Not Found");

            if (!symptomsInput || symptomsInput.value.trim() === "") {
                alert("Please enter symptoms before searching for medicines.");
                event.preventDefault();
            } else if (!ageInput || ageInput.value.trim() === "" || isNaN(ageInput.value) || parseInt(ageInput.value) <= 0) {
                alert("Please enter a valid age.");
                event.preventDefault();
            } else if (!genderInput || genderInput.value === "") {
                alert("Please select your gender.");
                event.preventDefault();
            }
        });
    }

    // First Aid Pages: Show alerts for critical situations
    let emergencyLinks = document.querySelectorAll(".emergency a");
    if (emergencyLinks) {
        emergencyLinks.forEach(link => {
            link.addEventListener("click", function (event) {
                let confirmRedirect = confirm("Are you sure you want to proceed to the First Aid Guide? Follow the steps carefully.");
                if (!confirmRedirect) {
                    event.preventDefault();
                }
            });
        });
    }

    // Image hover effect - Highlight emergency images
    let emergencyImages = document.querySelectorAll(".emergency img");
    emergencyImages.forEach(img => {
        img.addEventListener("mouseover", function () {
            this.style.transform = "scale(1.1)";
            this.style.transition = "transform 0.3s ease-in-out";
        });

        img.addEventListener("mouseout", function () {
            this.style.transform = "scale(1)";
        });
    });
});
