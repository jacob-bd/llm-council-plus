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

  // Groq (Provider, often Llama or Mixtral)
  // Check this BEFORE Meta/Mistral because Groq hosts those models
  if (id.includes('groq') || id.includes('versatile') || id.includes('instant')) {
    return { name: 'Groq', color: '#f97316', short: 'Groq', icon: '⚡' };
  }

  // Meta / Llama
  if (id.includes('meta') || id.includes('llama')) {
    return { name: 'Meta', color: '#0668e1', short: 'Llama', icon: '🦙' };
  }

  // DeepSeek
  if (id.includes('deepseek')) {
    return { name: 'DeepSeek', color: '#4e80ee', short: 'DeepSeek', icon: '🐋' };
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

  let name = modelId.replace(/:free$/, '');

  if (name.includes(':')) {
    name = name.split(':').slice(1).join(':');
  }

  if (name.includes('/')) {
    name = name.split('/').pop();
  }

  return name;
};
