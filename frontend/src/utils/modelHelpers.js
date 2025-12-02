// Helper to get visual properties for models

export const getModelVisuals = (modelId) => {
  if (!modelId) return { name: 'Unknown', color: '#94a3b8', short: '?' };

  const id = modelId.toLowerCase();

  // Ollama - CHECK FIRST because "ollama" contains "llama" substring
  if (id.startsWith('ollama:')) {
    return { name: 'Ollama', color: '#f1f5f9', short: 'Local', icon: '🦙' };
  }

  // OpenAI
  if (id.includes('openai') || id.includes('gpt')) {
    return { name: 'OpenAI', color: '#10a37f', short: 'GPT', icon: '🤖' };
  }
  
  // Anthropic
  if (id.includes('anthropic') || id.includes('claude')) {
    return { name: 'Anthropic', color: '#d97757', short: 'Claude', icon: '🧠' };
  }
  
  // Google
  if (id.includes('google') || id.includes('gemini')) {
    return { name: 'Google', color: '#4285f4', short: 'Gemini', icon: '✨' };
  }
  
  // Mistral
  if (id.includes('mistral')) {
    return { name: 'Mistral', color: '#5a4bda', short: 'Mistral', icon: '🌪️' };
  }
  
  // Meta / Llama
  if (id.includes('meta') || id.includes('llama')) {
    return { name: 'Meta', color: '#0668e1', short: 'Llama', icon: '🦙' };
  }
  
  // DeepSeek
  if (id.includes('deepseek')) {
    return { name: 'DeepSeek', color: '#4e80ee', short: 'DeepSeek', icon: '🐋' };
  }

  // Groq (Provider, often Llama or Mixtral)
  if (id.includes('groq')) {
    return { name: 'Groq', color: '#f97316', short: 'Groq', icon: '⚡' }; // Orange instead of red to avoid error-state appearance
  }

  // Local (fallback for models without provider prefix or slash)
  if (!id.includes('/') && !id.includes(':')) {
    return { name: 'Local', color: '#f1f5f9', short: 'Local', icon: '💻' };
  }

  // Default
  return { name: 'Model', color: '#94a3b8', short: 'AI', icon: '🤖' };
};

export const getShortModelName = (modelId) => {
  if (!modelId) return 'Unknown';
  // Handle "provider/model-name" format
  const parts = modelId.split('/');
  if (parts.length > 1) return parts[1];
  // Handle "provider:model-name" format
  const colParts = modelId.split(':');
  if (colParts.length > 1) return colParts[1];
  return modelId;
};
