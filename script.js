function predictDisease() {
  const result = document.getElementById("result");
  result.innerText = "Analyzing...";

  // Simulate prediction
  setTimeout(() => {
    result.innerText = "Prediction: Healthy Leaf (Simulated)";
  }, 2000);
}
