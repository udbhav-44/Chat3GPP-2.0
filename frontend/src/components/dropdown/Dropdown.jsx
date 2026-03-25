import React from "react";
import "./dropdown.css";

const Dropdown = ({
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
  modelOptions,
}) => {
  const handleProviderChange = (event) => {
    onProviderChange(event.target.value);
  };

  const handleModelChange = (event) => {
    onModelChange(event.target.value);
  };

  return (
    <div className="model-dropdown">
      <select
        value={selectedProvider}
        onChange={handleProviderChange}
        className="model-dropdown__select"
      >
        <option value="openai">OpenAI</option>
        <option value="deepseek">DeepSeek</option>
      </select>
      <select
        value={selectedModel}
        onChange={handleModelChange}
        className="model-dropdown__select"
      >
        {(modelOptions[selectedProvider] || []).map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </div>
  );
};

export default Dropdown;
