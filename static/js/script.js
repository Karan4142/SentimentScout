// static/js/script.js

function showLoadingPanel() {
    // Display the loading panel
    document.getElementById('loading-panel').style.display = 'block';

   
}

// Function to be called when the processing is done
function showChartContainer() {
    // Display the chart container
    document.getElementById('chart-container').style.display = 'block';

    document.getElementById('loading-panel').style.display = 'none';
}
