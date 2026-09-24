document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const btn = document.getElementById('submit-btn');
    const originalText = btn.innerText;
    btn.innerText = 'Predicting...';
    btn.disabled = true;

    // Gather data
    const formData = {
        state: document.getElementById('state').value,
        crop: document.getElementById('crop').value,
        area: document.getElementById('area').value,
        fertilizer: document.getElementById('fertilizer').value,
        pesticide: document.getElementById('pesticide').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (response.ok && data.success) {
            document.getElementById('result-section').style.display = 'block';
            
            // Primary Result
            document.getElementById('recommended-season').innerText = data.recommended_season;
            document.getElementById('predicted-yield').innerText = data.yield.toLocaleString();
            document.getElementById('predicted-production').innerText = data.production.toLocaleString();
            
            // Supporting - Conditions
            const conditionsList = document.getElementById('conditions-list');
            conditionsList.innerHTML = `
                <li><strong>Soil N:</strong> ${data.soil.N}</li>
                <li><strong>Soil P:</strong> ${data.soil.P}</li>
                <li><strong>Soil K:</strong> ${data.soil.K}</li>
                <li><strong>pH:</strong> ${data.soil.pH}</li>
                <li><strong>Temperature:</strong> ${data.weather.avg_temp_c.toFixed(1)} °C</li>
                <li><strong>Rainfall:</strong> ${data.weather.total_rainfall_mm.toFixed(1)} mm</li>
                <li><strong>Humidity:</strong> ${data.weather.avg_humidity_percent.toFixed(1)} %</li>
            `;
            
            // Supporting - Season Comparison
            const seasonBody = document.getElementById('season-comparison-body');
            seasonBody.innerHTML = '';
            data.season_comparison.forEach(item => {
                const tr = document.createElement('tr');
                const tdSeason = document.createElement('td');
                tdSeason.innerText = item.season;
                const tdYield = document.createElement('td');
                tdYield.innerText = item.yield.toLocaleString();
                tr.appendChild(tdSeason);
                tr.appendChild(tdYield);
                seasonBody.appendChild(tr);
            });
            
            document.getElementById('model-used').innerText = data.model_used;
            
            // Scroll to results
            document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + (data.error || 'Something went wrong'));
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to the server.');
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});
