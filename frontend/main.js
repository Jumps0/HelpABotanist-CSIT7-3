
//   ----------  Leaflet Map Setup ---------

// Initialize map
const map = L.map('map').setView([56.2639, 9.5018], 6);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors',
  maxZoom: 18,
}).addTo(map);

//      ---------   For plant dropdown ---------------
 
let selectedPlant = null;
let selectedModel = null;
let plantNames = []; // Array to store plant names from the header row
let gridSquares = []; // Global array to store grid squares for clearing later

// Function to fetch and parse plant data from CSV
async function fetchPlantData() {
  try {
    const response = await fetch('datagrid.csv');
    const text = await response.text();
    const rows = text.trim().split('\n');

    // Extract header row and plant names
    const headerRow = rows[0].split(',');
    plantNames = headerRow.slice(12).map(name => name.trim());
    console.log("Plant Names:", plantNames);

    // Populate dropdown with plant names
    const dropdown = document.getElementById('dropdown');
    dropdown.innerHTML = ''; // Clear previous items
    plantNames.sort((a, b) => a.localeCompare(b)); // Sort alphabetically
    plantNames.forEach(plant => {
      const item = document.createElement('div');
      item.className = 'dropdown-item';
      item.textContent = plant;
      item.dataset.plant = plant;
      dropdown.appendChild(item);
    });

    // Parse CSV data into an array of objects
    const data = rows.slice(1).map(row => {
      const values = row.split(',');
      const obj = {};
      headerRow.forEach((key, index) => {
        obj[key.trim()] = values[index] ? values[index].trim() : null;
      });
      return obj;
    });

													   
    return data;
  } catch (error) {
    console.error("Error fetching or parsing CSV data:", error);
  }
}


//       --------- Clear HeatMap    ------------


// Clear existing grid squares
function clearGrid() {
  gridSquares.forEach(square => map.removeLayer(square));
  gridSquares = [];
  // Clear the prediction results
  const predictionResultElement = document.getElementById('prediction-result');
  if (predictionResultElement) {
    predictionResultElement.innerHTML = ' Choose a plant to see predictions.\n Prediction takes some time until please wait.'; // Clear any displayed content
  }
}

//       --------- HeatMap  as a grid and color defining for Occurrence  ------------

let occurrenceLayer = L.layerGroup().addTo(map);
let predictionLayer = L.layerGroup().addTo(map);


// Generate grid cells based on occurrences of the selected plant
function generateGrid(flowerColumn, flowerData) {
  clearGrid();
  const gridSize = 0.2; // Grid cell size in degrees
  const gridCounts = {}; // Store counts for each grid cell

 

  // Loop through flower data and calculate occurrences for each grid cell
  flowerData.forEach(row => {
    const lat = parseFloat(row.latitude);
    const lon = parseFloat(row.longitude);
    const occurrence = parseFloat(row[flowerColumn]);

    if (!isNaN(lat) && !isNaN(lon) && !isNaN(occurrence) && occurrence >= 0) {
      // Determine the grid cell based on lat and lon
      const latGrid = Math.floor(lat / gridSize) * gridSize;
      const lonGrid = Math.floor(lon / gridSize) * gridSize;
      const cellKey = `${latGrid},${lonGrid}`; // Unique key for the grid cell
      gridCounts[cellKey] = (gridCounts[cellKey] || 0) + occurrence;
    }
  });

							  
			  

  // Create grid cells with color coding based on occurrences
  Object.keys(gridCounts).forEach(cellKey => {
    const [latGrid, lonGrid] = cellKey.split(',').map(Number);
    const count = gridCounts[cellKey];

    // Define color based on count (green to red scale)
    const color = count === 0 ? 'blue' :
                  count <= 2 ? 'green' :
                  count <= 5 ? 'yellow' :
                  count <= 8 ? 'orange' :
                  'red';
    const square = L.polygon(
       [
          [latGrid, lonGrid],                       // Bottom-left
          [latGrid + gridSize, lonGrid],            // Bottom-right
          [latGrid + gridSize, lonGrid + gridSize], // Top-right
          [latGrid, lonGrid + gridSize],            // Top-left
          [latGrid, lonGrid],                       // Close the square
       ],
       {
          color: 'black', // Border color
          weight: 0.9,    // Border thickness
          fillColor: color, // Dynamic fill color
          fillOpacity: 0.7, // Opacity of the square fill
       }
      );
                  // Add a popup with information about the grid cell
      const popupMessage = `
        <strong>Grid Cell Info:</strong><br>
        Latitude: ${latGrid.toFixed(2)} to ${(latGrid + gridSize).toFixed(2)}<br>
        Longitude: ${lonGrid.toFixed(2)} to ${(lonGrid + gridSize).toFixed(2)}<br>
        Occurrences: ${count}`;
      square.bindPopup(popupMessage);
      
      // Add the square to the map
      square.addTo(occurrenceLayer);
      gridSquares.push(square); // Store the square for later removal
                  
    });
}

//     -----------  Creating Occurrence button and connecting with CSV file ----------

// Handle plant selection and generate grid
function setupSearch(data) {
  clearGrid();
  const searchButton = document.getElementById('search-button');
  searchButton.addEventListener('click', () => {
    searchButton.classList.add('active-button');
    if (!selectedPlant) {
      alert('Please select a plant.');
      return;
    }
    
     
    // Filter data for the selected plant with presence >= 1
    const filteredData = data.filter(row => {
      const presence = parseInt(row[selectedPlant], 10);
      return !isNaN(presence) && presence >= 0;
    });

     // Generate the grid based on occurrences
    generateGrid(selectedPlant, filteredData);

    // Calculate the sum of all presences
    const presenceSum = filteredData.reduce((sum, row) => {
      const presence = parseInt(row[selectedPlant], 10);
      return sum + (isNaN(presence) ? 0 : presence);
    }, 0);

    // Update the result message
    const resultElement = document.getElementById('prediction-result');
    resultElement.innerHTML = `Total number of presence from the records for <strong>${selectedPlant}</strong>: <strong>${presenceSum}</strong>`;
   
  });
}

//     -----------  Creating Prediction button and connecting with backend file ----------


function selectMethod(method) {
  const input = document.getElementById('model-search-input');
  input.value = method; // Set the selected method in the input field
  document.getElementById('model-dropdown').style.display = 'none'; // Hide the dropdown
  selectedModel = method; // Store the selected model globally
}
console.log(`Selected plant: ${selectedPlant}, Selected model: ${selectedModel}`);
async function setupPredictionButton() {
  clearGrid();
  const predictionButton = document.getElementById('predictionButton');
  const modelSearchInput = document.getElementById('model-search-input');
  const plantSearchInput = document.getElementById('search-input');
  
  
  // Enable/disable button when both plant and model are selected
  plantSearchInput.addEventListener('input', function(event) {
    selectedPlant = event.target.value.trim(); // Ensure it trims any spaces
    console.log('Plant updated:', selectedPlant); // Debug log to verify plant selection
    checkSelection(); // Check button status
  });

  modelSearchInput.addEventListener('input', function(event) {
    selectedModel = event.target.value.trim(); // Ensure it trims any spaces
    console.log('Model updated:', selectedModel); // Debug log to verify the selected model
    checkSelection(); // Check button status
  });
 
    // Debug log for checking selected values
  console.log(`Selected plant: ${selectedPlant}, Selected model: ${selectedModel}`);
  function checkSelection() {
    

    // Make sure both plant and model are selected and have values
    if (selectedPlant && selectedModel) {
      predictionButton.disabled = false; // Enable button when both are selected
    } else {
      predictionButton.disabled = true; // Disable button if either is not selected
    }
  }
  predictionButton.addEventListener('click', async () => {
    console.log("Prediction Button clicked"); // Debug log
    clearGrid();
    predictionButton.classList.add('active-button');

    if (!selectedPlant || !selectedModel) {
      alert('Please select both a plant and a model.');
      return;
    }
    // Map the selected model to the appropriate endpoint
    const modelEndpoints = {
      "Label Propagation": "/predict_LP",
      "KNeighbors Classifier": "/predict_knn",
      "Random Forest Classifier": "/predict_RF",
      "Gradient Boosting Classifier": "/predict_GBC",
      
    };
    const endpoint = modelEndpoints[selectedModel];
    if (!endpoint) {
      alert('Invalid model selected.');
      return;
    }

    console.log(`Selected Endpoint: ${endpoint}`);
    console.log("Prediction Button clicked, connecting to API...");
  
//  -----------  Label Propagation -------------
    try {
      // Send a POST request to FastAPI server
      const requestData = { plant: selectedPlant};

      console.log('Sending request to the server...', requestData);
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData), 
      });

      console.log("Selected @Plant:", selectedPlant);
      console.log("Selected Model:", selectedModel);
      console.log("Request Data:", requestData);
       
      // Handle the response
      if (!response.ok) {
        throw new Error(`Failed to fetch prediction DATA: ${response.statusText}`);
      }

      console.log("I'm going to get JSON data ")
      // Handle the response from the backend
      const result = await response.json();
      console.log("Received response:",result); // You can log to check the response
     
      if (!result.heatmap || !Array.isArray(result.heatmap)|| result.heatmap.length === 0) {
        console.error("Invalid heatmap data:", result.heatmap);
        alert("No valid heatmap data received.");
        return;
      }
      
      // Display the heatmap
      displayPredictionHeatmap(result.heatmap, result.trainNodes, result.testNodes);

      // Show the custom control for train, test, and heatmap checkboxes
      createCheckboxControl(); // Display the control with checkboxe
      
      //To display a prediction number (e.g., prediction count or a similar metric)
      const predictionNumber = result.prediction|| 0;
      const resultElement = document.getElementById('prediction-result');
    
      resultElement.innerHTML = `Prediction of the accurracy for the plant <strong>${selectedPlant}</strong> in  the <strong>${selectedModel} model </strong> is : <strong>${predictionNumber}%</strong>`;

    } catch (error) {
       console.error('Prediction request failed:', error);
       alert(`Failed to fetch prediction data: ${error.message}`);
     }
    });
    
 }
    

//   ------- Creating Prediction HeatMap with Train & Test data ----------------- 


// Global layer variables for prediction heatmap, train, and test data

let trainLayer = L.layerGroup().addTo(map);
let testLayer = L.layerGroup().addTo(map);
// Function to create custom control for checkboxes
// Function to create custom control for checkboxes
function createCheckboxControl() {
  const control = L.Control.extend({
    options: {
      position: 'topright',
    },
    onAdd: function () {
      const div = L.DomUtil.create('div', 'leaflet-control-checkbox');
      clearGrid();
      
      // Create checkboxes and labels
      const trainCheckbox = document.createElement('input');
      trainCheckbox.type = 'checkbox';
      trainCheckbox.id = 'trainCheckbox';
      trainCheckbox.checked = true; // Initially unchecked
      const trainLabel = document.createElement('label');
      trainLabel.textContent = 'Train Data';
      trainLabel.setAttribute('for', 'trainCheckbox');
      
      const testCheckbox = document.createElement('input');
      testCheckbox.type = 'checkbox';
      testCheckbox.id = 'testCheckbox';
      testCheckbox.checked = true; // Initially unchecked
      const testLabel = document.createElement('label');
      testLabel.textContent = 'Test Data';
      testLabel.setAttribute('for', 'testCheckbox');
      
      const heatmapCheckbox = document.createElement('input');
      heatmapCheckbox.type = 'checkbox';
      heatmapCheckbox.id = 'heatmapCheckbox';
      heatmapCheckbox.checked = true; // Initially checked
      const heatmapLabel = document.createElement('label');
      heatmapLabel.textContent = 'Prediction Heatmap';
      heatmapLabel.setAttribute('for', 'heatmapCheckbox');
      
      // Append checkboxes to the control
      div.appendChild(trainCheckbox);
      div.appendChild(trainLabel);
      div.appendChild(document.createElement('br'));
      div.appendChild(testCheckbox);
      div.appendChild(testLabel);
      div.appendChild(document.createElement('br'));
      div.appendChild(heatmapCheckbox);
      div.appendChild(heatmapLabel);

      // Event listeners for checkbox changes
      trainCheckbox.addEventListener('change', function () {
        if (trainCheckbox.checked) {
          map.addLayer(trainLayer); // Show Train Data
        } else {
          map.removeLayer(trainLayer); // Hide Train Data
        }
      });

      testCheckbox.addEventListener('change', function () {
        if (testCheckbox.checked) {
          map.addLayer(testLayer); // Show Test Data
        } else {
          map.removeLayer(testLayer); // Hide Test Data
        }
      });

      heatmapCheckbox.addEventListener('change', function () {
        if (heatmapCheckbox.checked) {
          map.addLayer(predictionLayer); // Show Prediction Heatmap
        } else {
          map.removeLayer(predictionLayer); // Hide Prediction Heatmap
        }
      });

      return div;
    }
  });

  map.addControl(new control());
}

//    ------------ Displaying Prediction Heatmap  -----------------

function displayPredictionHeatmap(heatmap) {

  predictionLayer.clearLayers();

								
						   

  console.log("Preparing HeatMap");

 if (!Array.isArray(heatmap)) {
    console.error("Invalid or missing heatmap data:", heatmap);
    return;
  }	


  const gridSize = 0.2; // Define grid size
  const values = heatmap.map(item => item.intensity);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);

  const colorScale = chroma.scale(['#f7786b', '#7b3c35']).domain([minValue, maxValue]);

  // Display heatmap
  heatmap.forEach(({ latitude, longitude, intensity, isTrain, isTest }) => {
    if (latitude === undefined || longitude === undefined) return;

    const color = colorScale(intensity).hex();
    const square = L.polygon(
      [
        [latitude, longitude],
        [latitude + gridSize, longitude],
        [latitude + gridSize, longitude + gridSize],
        [latitude, longitude + gridSize],
        [latitude, longitude],
      ],
      {
        color: 'black',
        weight: 0.9,
        fillColor: color,
        fillOpacity: 0.6,
      }
    );

    square.bindPopup(`Prediction Info:<br>
      Latitude: ${latitude.toFixed(2)}<br>
      Longitude: ${longitude.toFixed(2)}<br>
      Intensity: ${intensity.toFixed(2)}`);

    square.addTo(predictionLayer);

    // Add Train and Test markers based on conditions
    if (isTrain) {
      L.circleMarker([latitude, longitude], {
        radius: 2,
        fillColor: 'greenyellow',
        fillOpacity: 0.6,
        color: 'black',
        weight: 1,
      }).addTo(trainLayer);
      square.bindPopup(`Prediction Info:<br>
        Latitude: ${latitude.toFixed(2)}<br>
        Longitude: ${longitude.toFixed(2)}<br>
        `);
  
      square.addTo(predictionLayer);
    }

    if (isTest) {
      L.circleMarker([latitude, longitude], {
        radius: 4,
        fillColor: 'red',
        fillOpacity: 0.6,
        color: 'black',
        weight: 1,
      }).addTo(testLayer);
      square.bindPopup(`Prediction Info:<br>
        Latitude: ${latitude.toFixed(2)}<br>
        Longitude: ${longitude.toFixed(2)}<br>
        `);
  
      square.addTo(predictionLayer);
    }
  });
}
// Function to toggle color descriptions
function toggleColorDescription(type) {
  const occurrenceDescription = document.getElementById('occurrence-color-description');
  const predictionDescription = document.getElementById('prediction-color-description');

  if (type === 'occurrence') {
    occurrenceDescription.style.display = 'block';
    predictionDescription.style.display = 'none';
  } else if (type === 'prediction') {
    occurrenceDescription.style.display = 'none';
    predictionDescription.style.display = 'block';
  } else {
    occurrenceDescription.style.display = 'none';
    predictionDescription.style.display = 'none';
  }
}

// Modify search button to show occurrence legend
const searchButton = document.getElementById('search-button');
searchButton.addEventListener('click', () => {
  
  toggleColorDescription('occurrence'),
 
  setActiveButton(e.target); 
});

// Modify prediction button to show prediction legend
predictionButton.addEventListener('click', () => {
 
  toggleColorDescription('prediction'),
 
  setActiveButton(e.target);
});

// Hide both legends initially
toggleColorDescription(null);

//    --------- Assigning Functions for Occurrence and Prediction buttons  -------------

document.getElementById('predictionButton').addEventListener('click', async () => {
 clearGrid();
  const heatmap = await setupPredictionButton(); // Fetch prediction data from the backend
  if (heatmap) { 
  displayPredictionHeatmap(heatmap); // Display the heatmap
  }
  else {
    console.error("No heatmap data received.");
  }
});

function setupDropdown() {
    const searchInput = document.getElementById('search-input');
    const dropdown = document.getElementById('dropdown');
  
    // Show dropdown on focus
    searchInput.addEventListener('focus', () => {
      dropdown.classList.add('show');
    });
  
    // Filter dropdown items based on user input
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      Array.from(dropdown.children).forEach(item => {
        item.style.display = item.textContent.toLowerCase().includes(query) ? 'block' : 'none';
      });
    });
  
    // Handle plant selection from dropdown
    dropdown.addEventListener('click', (e) => {
      if (e.target.classList.contains('dropdown-item')) {
        selectedPlant = e.target.dataset.plant;
        searchInput.value = selectedPlant;
        dropdown.classList.remove('show');
      }
    });
    // Hide dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!dropdown.contains(e.target) && !searchInput.contains(e.target)) {
        dropdown.classList.remove('show');
      }
    });
  }

  //  ----------   ML model dropdown   ---------------

  function toggleDropdown() {
    const dropdown = document.getElementById('model-dropdown');
    dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
}



// Close ML model dropdown when clicking outside
document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('model-dropdown');
    const searchInput1 = document.getElementById('model-search-input');
    if (!dropdown.contains(e.target) && !searchInput1.contains(e.target)) {
        dropdown.style.display = 'none';
    }
});

// Example JavaScript to toggle the active button class
const buttons = document.querySelectorAll('.btn-group button');

buttons.forEach(button => {
  button.addEventListener('click', () => {
    buttons.forEach(btn => btn.classList.remove('active-button')); // Remove the active class from all buttons
    button.classList.add('active-button'); // Add active class to the clicked button
  });
});

//   -------------  Initializing overall functions  ------------------

// Fetch and display plant data, set up dropdown and search functionality
async function initializeData() {
  const data = await fetchPlantData(); // Fetch and parse CSV
  setupDropdown(); // Setup dropdown functionality
  setupSearch(data); // Setup search and heatmap rendering
  setupPredictionButton(); // event listener for prediction button
}

// Initialize the app
initializeData();
