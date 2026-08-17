import { useEffect, useState } from "react";

function App() {
  const [health, setHealth] = useState({ status: "loading", service: "backend" });
  const [flows, setFlows] = useState([]);
  const [runtimeConfig, setRuntimeConfig] = useState(null);
  const [questionnaire, setQuestionnaire] = useState(null);
  const [error, setError] = useState("");
  const [answers, setAnswers] = useState({});

  useEffect(() => {
    async function fetchHealth() {
      const [healthResponse, questionnaireStatusResponse, sttResponse, ttsResponse, predictionResponse, schemaResponse, runtimeResponse] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/flows/questionnaire/status"),
        fetch("/api/flows/stt/status"),
        fetch("/api/flows/tts/status"),
        fetch("/api/flows/prediction/status"),
        fetch("/api/flows/questionnaire/schema"),
        fetch("/api/runtime/config")
      ]);

      const responses = [
        healthResponse,
        questionnaireStatusResponse,
        sttResponse,
        ttsResponse,
        predictionResponse,
        schemaResponse,
        runtimeResponse
      ];
      for (const response of responses) {
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
      }

      setHealth(await healthResponse.json());
      const loadedSchema = await schemaResponse.json();
      const loadedRuntime = await runtimeResponse.json();

      setQuestionnaire(loadedSchema);
      setRuntimeConfig(loadedRuntime);
      setFlows([
        await questionnaireStatusResponse.json(),
        await sttResponse.json(),
        await ttsResponse.json(),
        await predictionResponse.json()
      ]);

      const initialAnswers = {};
      for (const question of loadedSchema.questions ?? []) {
        if (question.type === "select" && Array.isArray(question.options) && question.options.length > 0) {
          initialAnswers[question.id] = question.options[0].value;
        } else {
          initialAnswers[question.id] = "";
        }
      }
      setAnswers(initialAnswers);
    }

    fetchHealth().catch((err) => {
      setError(err instanceof Error ? err.message : "Unknown error");
    });
  }, []);

  const criticalityScore = (() => {
    let score = 0;
    for (const question of questionnaire?.questions ?? []) {
      if (question.type !== "select" || !Array.isArray(question.options)) {
        continue;
      }
      const selectedValue = answers[question.id];
      const selectedOption = question.options.find((option) => option.value === selectedValue);
      if (selectedOption?.risk_weight) {
        score += Number(selectedOption.risk_weight);
      }
    }
    return score;
  })();

  return (
    <main className="app">
      <h1>Non-invasive Anaemia and Malnutrition Detection</h1>
      <p>Hackathon scaffold: questionnaire + STT/TTS + prediction flows are independent.</p>
      {error ? (
        <p className="error">Backend error: {error}</p>
      ) : (
        <>
          <p>
            Backend health: <strong>{health.status}</strong> ({health.service})
          </p>
          <ul>
            {flows.map((flow) => (
              <li key={flow.flow}>
                {flow.flow}: <strong>{flow.status}</strong>
              </li>
            ))}
          </ul>
          <p>
            Runtime mode: <strong>{runtimeConfig?.runtime_mode ?? "unknown"}</strong>
          </p>
          <p>
            Audio execution: STT ({runtimeConfig?.audio_execution?.stt_mode}) / TTS (
            {runtimeConfig?.audio_execution?.tts_mode})
          </p>
          <h2>{questionnaire?.title ?? "Questionnaire"}</h2>
          {(questionnaire?.questions ?? []).map((question) => (
            <label key={question.id}>
              {question.label}:
              {question.type === "select" ? (
                <select
                  value={answers[question.id] ?? ""}
                  onChange={(event) =>
                    setAnswers((prev) => ({
                      ...prev,
                      [question.id]: event.target.value
                    }))
                  }
                >
                  {(question.options ?? []).map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={question.type === "number" ? "number" : "text"}
                  value={answers[question.id] ?? ""}
                  onChange={(event) =>
                    setAnswers((prev) => ({
                      ...prev,
                      [question.id]: event.target.value
                    }))
                  }
                />
              )}
            </label>
          ))}
          <p>STT/TTS hooking will be handled by your harness layer.</p>
          <p>
            Criticality score (placeholder): <strong>{criticalityScore}/100</strong>
          </p>
        </>
      )}
    </main>
  );
}

export default App;
