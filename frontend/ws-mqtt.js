async function loadConfig() {
  const resp = await fetch("mqtt.json");
  if (!resp.ok) throw new Error("Could not load mqtt.json");
  const config = await resp.json();
  return config;
}

loadConfig().then(config => {
  const ip = config.host;
  const port = config.port;
  const topic = config.topic;
  const brokerUrl = `ws://${ip}:${port}`;

  // Connect only after config is loaded
  const client = mqtt.connect(brokerUrl);

  client.on("connect", () => {
    console.log("Connected to MQTT broker");
  });

  client.on("error", (err) => {
    console.error("Connection error:", err);
  });

  // Enable the send button now that client is ready
  document.getElementById("sendBtn").onclick = () => {
    if (currentWalk.length === 0) {
      console.error("Current walk is empty! Not sending");
      return;
    }

    const walk = [...currentWalk];
    if (walk.length !== 0) {
      walk.pop();
    }
    const payload = JSON.stringify(walk);

    client.publish(topic, payload, {}, (err) => {
      if (err) {
        console.error("Publish failed: ", err);
      } else {
        console.log("Message sent");
      }
    });
  };
}).catch(err => {
  console.error("Failed to load config:", err);
});
