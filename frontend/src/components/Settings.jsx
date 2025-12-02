import { useState, useEffect } from 'react';
import { api } from '../api';
import './Settings.css';

const SEARCH_PROVIDERS = [
  {
    id: 'duckduckgo',
    name: 'DuckDuckGo',
    description: 'News search. Fast and free.',
    requiresKey: false,
    keyType: null,
  },
  {
    id: 'tavily',
    name: 'Tavily',
    description: 'Purpose-built for LLMs. Returns rich, relevant content. Requires API key.',
    requiresKey: true,
    keyType: 'tavily',
  },
  {
    id: 'brave',
    name: 'Brave Search',
    description: 'Privacy-focused search. 2,000 free queries/month. Requires API key.',
    requiresKey: true,
    keyType: 'brave',
  },
];

const DIRECT_PROVIDERS = [
  { id: 'openai', name: 'OpenAI', key: 'openai_api_key' },
  { id: 'anthropic', name: 'Anthropic', key: 'anthropic_api_key' },
  { id: 'google', name: 'Google', key: 'google_api_key' },
  { id: 'mistral', name: 'Mistral', key: 'mistral_api_key' },
  { id: 'deepseek', name: 'DeepSeek', key: 'deepseek_api_key' },
];

export default function Settings({ onClose, ollamaStatus, onRefreshOllama, initialSection = 'llm_keys' }) {
  const [activeSection, setActiveSection] = useState(initialSection); // 'llm_keys', 'council', 'prompts', 'search', 'import_export'

  const [settings, setSettings] = useState(null);
  const [selectedSearchProvider, setSelectedSearchProvider] = useState('duckduckgo');
  const [fullContentResults, setFullContentResults] = useState(3);

  // OpenRouter State
  const [openrouterApiKey, setOpenrouterApiKey] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [isTestingOpenRouter, setIsTestingOpenRouter] = useState(false);
  const [openrouterTestResult, setOpenrouterTestResult] = useState(null);

  // Groq State
  const [groqApiKey, setGroqApiKey] = useState('');
  const [isTestingGroq, setIsTestingGroq] = useState(false);
  const [groqTestResult, setGroqTestResult] = useState(null);

  // Ollama State
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState('http://localhost:11434');
  const [ollamaAvailableModels, setOllamaAvailableModels] = useState([]);
  const [isTestingOllama, setIsTestingOllama] = useState(false);
  const [ollamaTestResult, setOllamaTestResult] = useState(null);

  // Direct Provider State
  const [directKeys, setDirectKeys] = useState({
    openai_api_key: '',
    anthropic_api_key: '',
    google_api_key: '',
    mistral_api_key: '',
    deepseek_api_key: ''
  });
  const [directAvailableModels, setDirectAvailableModels] = useState([]);

  // Validation State
  const [validatingKeys, setValidatingKeys] = useState({});
  const [keyValidationStatus, setKeyValidationStatus] = useState({});

  // Search API Keys
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [braveApiKey, setBraveApiKey] = useState('');
  const [isTestingTavily, setIsTestingTavily] = useState(false);
  const [isTestingBrave, setIsTestingBrave] = useState(false);
  const [tavilyTestResult, setTavilyTestResult] = useState(null);
  const [braveTestResult, setBraveTestResult] = useState(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Enabled Providers (which sources are available)
  const [enabledProviders, setEnabledProviders] = useState({
    openrouter: true,
    ollama: false,
    direct: false  // Master toggle for all direct connections
  });

  // Individual direct provider toggles
  const [directProviderToggles, setDirectProviderToggles] = useState({
    openai: false,
    anthropic: false,
    google: false,
    mistral: false,
    deepseek: false
  });

  // Council Configuration (unified across all providers)
  const [councilModels, setCouncilModels] = useState([]);
  const [chairmanModel, setChairmanModel] = useState('');

  // Web Search Query Generator
  const [searchQueryModel, setSearchQueryModel] = useState('');

  // System Prompts State
  const [prompts, setPrompts] = useState({
    stage1_prompt: '',
    stage2_prompt: '',
    stage3_prompt: '',
    title_prompt: '',
    search_query_prompt: ''
  });
  const [activePromptTab, setActivePromptTab] = useState('stage1');

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showFreeOnly, setShowFreeOnly] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Remote/Local filter toggles per model type
  const [searchQueryFilter, setSearchQueryFilter] = useState('remote');  // 'remote' or 'local'
  const [councilMemberFilters, setCouncilMemberFilters] = useState({});  // Per-member filters (indexed by member index)
  const [chairmanFilter, setChairmanFilter] = useState('remote');

  useEffect(() => {
    loadSettings();
  }, []);

  // Update activeSection when initialSection prop changes
  useEffect(() => {
    setActiveSection(initialSection);
  }, [initialSection]);

  // Check for changes
  useEffect(() => {
    if (!settings) return;

    const checkChanges = () => {
      if (selectedSearchProvider !== settings.search_provider) return true;
      if (fullContentResults !== (settings.full_content_results ?? 3)) return true;

      // Enabled Providers
      if (JSON.stringify(enabledProviders) !== JSON.stringify(settings.enabled_providers)) return true;
      if (JSON.stringify(directProviderToggles) !== JSON.stringify(settings.direct_provider_toggles)) return true;

      // Council Configuration (unified)
      if (JSON.stringify(councilModels) !== JSON.stringify(settings.council_models)) return true;
      if (chairmanModel !== settings.chairman_model) return true;

      // Remote/Local filters
      if (JSON.stringify(councilMemberFilters) !== JSON.stringify(settings.council_member_filters || {})) return true;
      if (chairmanFilter !== (settings.chairman_filter || 'remote')) return true;
      if (searchQueryFilter !== (settings.search_query_filter || 'remote')) return true;

      // Web Search Query Generator
      if (searchQueryModel !== settings.search_query_model) return true;

      // Prompts
      if (prompts.stage1_prompt !== settings.stage1_prompt) return true;
      if (prompts.stage2_prompt !== settings.stage2_prompt) return true;
      if (prompts.stage3_prompt !== settings.stage3_prompt) return true;
      if (prompts.search_query_prompt !== settings.search_query_prompt) return true;

      // Note: API keys are auto-saved on test, so we don't check them here

      return false;
    };

    setHasChanges(checkChanges());
  }, [
    settings,
    selectedSearchProvider,
    fullContentResults,
    enabledProviders,
    directProviderToggles,
    councilModels,
    chairmanModel,
    councilMemberFilters,
    chairmanFilter,
    searchQueryFilter,
    searchQueryModel,
    prompts
  ]);

  // Helper to determine if filters need to switch based on availability
  const isRemoteAvailable = enabledProviders.openrouter || enabledProviders.direct || enabledProviders.groq;
  const isLocalAvailable = enabledProviders.ollama;

  const getNewFilter = (currentFilter) => {
    if (currentFilter === 'remote' && !isRemoteAvailable && isLocalAvailable) return 'local';
    if (currentFilter === 'local' && !isLocalAvailable && isRemoteAvailable) return 'remote';
    return currentFilter;
  };

  // Effect 1: Auto-update Council Member filters when providers change or members are added
  useEffect(() => {
    setCouncilMemberFilters(prev => {
      const next = { ...prev };
      let changed = false;
      // Check all council member indices
      for (let i = 0; i < councilModels.length; i++) {
        const currentFilter = next[i] || 'remote'; // Default is 'remote'
        const newFilter = getNewFilter(currentFilter);
        if (newFilter !== currentFilter) {
          next[i] = newFilter;
          changed = true;
          // Clear model if filter changed to force re-selection
          handleCouncilModelChange(i, '');
        }
      }
      return changed ? next : prev;
    });
  }, [enabledProviders, councilModels.length]);

  // Effect 2: Auto-update Chairman and Search filters when providers change
  // Note: We intentionally exclude councilModels.length to prevent resetting these when adding members
  useEffect(() => {
    // Update Chairman
    const newChairmanFilter = getNewFilter(chairmanFilter);
    if (newChairmanFilter !== chairmanFilter) {
      setChairmanFilter(newChairmanFilter);
      setChairmanModel('');
    }

    // Update Search Query Model
    const newSearchFilter = getNewFilter(searchQueryFilter);
    if (newSearchFilter !== searchQueryFilter) {
      setSearchQueryFilter(newSearchFilter);
      setSearchQueryModel('');
    }
  }, [enabledProviders, chairmanFilter, searchQueryFilter]);

  const loadSettings = async () => {
    try {
      const data = await api.getSettings();

      // Set settings immediately to show UI
      setSettings(data);

      setSelectedSearchProvider(data.search_provider || 'duckduckgo');
      setFullContentResults(data.full_content_results ?? 3);

      // Enabled Providers - use saved settings if available, otherwise auto-enable based on configured keys
      if (data.enabled_providers) {
        // User has explicitly set their preferences - use them
        setEnabledProviders(data.enabled_providers);
      } else {
        // First time or no saved preferences - auto-enable based on what's configured
        const hasDirectConfigured = !!(data.openai_api_key_set || data.anthropic_api_key_set ||
          data.google_api_key_set || data.mistral_api_key_set || data.deepseek_api_key_set);

        setEnabledProviders({
          openrouter: !!data.openrouter_api_key_set || (!hasDirectConfigured && !ollamaStatus?.connected && !data.groq_api_key_set),
          ollama: ollamaStatus?.connected || false,
          groq: !!data.groq_api_key_set,
          direct: hasDirectConfigured
        });
      }

      // Individual direct provider toggles - load from saved settings
      if (data.direct_provider_toggles) {
        setDirectProviderToggles(data.direct_provider_toggles);
      } else {
        // Fallback for first-time users: auto-enable if API key is configured
        setDirectProviderToggles({
          openai: !!data.openai_api_key_set,
          anthropic: !!data.anthropic_api_key_set,
          google: !!data.google_api_key_set,
          mistral: !!data.mistral_api_key_set,
          deepseek: !!data.deepseek_api_key_set
        });
      }

      // Council Configuration (unified)
      setCouncilModels(data.council_models || []);
      setChairmanModel(data.chairman_model || '');

      // Remote/Local filters - load from saved settings
      if (data.council_member_filters) {
        setCouncilMemberFilters(data.council_member_filters);
      }
      if (data.chairman_filter) {
        setChairmanFilter(data.chairman_filter);
      }
      if (data.search_query_filter) {
        setSearchQueryFilter(data.search_query_filter);
      }

      // Web Search Query Generator
      setSearchQueryModel(data.search_query_model || 'google/gemini-2.5-flash');

      // Ollama Settings
      setOllamaBaseUrl(data.ollama_base_url || 'http://localhost:11434');

      // Prompts
      setPrompts({
        stage1_prompt: data.stage1_prompt || '',
        stage2_prompt: data.stage2_prompt || '',
        stage3_prompt: data.stage3_prompt || '',
        search_query_prompt: data.search_query_prompt || ''
      });

      // Clear Direct Keys (for security)
      setDirectKeys({
        openai_api_key: '',
        anthropic_api_key: '',
        google_api_key: '',
        mistral_api_key: '',
        deepseek_api_key: ''
      });
      setGroqApiKey(''); // Clear Groq key too

      // Load available models in background
      loadModels();
      loadOllamaModels(data.ollama_base_url || 'http://localhost:11434');

    } catch (err) {
      console.error("Error loading settings:", err);
      setError('Failed to load settings');
    }
  };

  const loadModels = async () => {
    setIsLoadingModels(true);
    try {
      const data = await api.getModels();
      if (data.models && data.models.length > 0) {
        // Sort models alphabetically
        const sorted = data.models.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        setAvailableModels(sorted);
      }

      // Fetch direct models from backend
      try {
        const directModels = await api.getDirectModels();
        setDirectAvailableModels(directModels);
      } catch (error) {
        console.error('Failed to fetch direct models:', error);
        // Fallback to empty list or basic models if fetch fails
        setDirectAvailableModels([]);
      }

    } catch (err) {
      console.warn('Failed to load models:', err);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const loadOllamaModels = async (baseUrl) => {
    try {
      const data = await api.getOllamaModels(baseUrl);
      if (data.models && data.models.length > 0) {
        // Sort models alphabetically
        const sorted = data.models.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        setOllamaAvailableModels(sorted);
      }
    } catch (err) {
      console.warn('Failed to load Ollama models:', err);
    }
  };


  const handleTestTavily = async () => {
    if (!tavilyApiKey) {
      setTavilyTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingTavily(true);
    setTavilyTestResult(null);
    try {
      const result = await api.testTavilyKey(tavilyApiKey);
      setTavilyTestResult(result);

      // Auto-save API key if validation succeeds
      if (result.success) {
        await api.updateSettings({ tavily_api_key: tavilyApiKey });
        setTavilyApiKey(''); // Clear input after save

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setTavilyTestResult({ success: false, message: 'Test failed' });
    } finally {
      setIsTestingTavily(false);
    }
  };

  const handleTestBrave = async () => {
    if (!braveApiKey) {
      setBraveTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingBrave(true);
    setBraveTestResult(null);
    try {
      const result = await api.testBraveKey(braveApiKey);
      setBraveTestResult(result);

      // Auto-save API key if validation succeeds
      if (result.success) {
        await api.updateSettings({ brave_api_key: braveApiKey });
        setBraveApiKey(''); // Clear input after save

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setBraveTestResult({ success: false, message: 'Test failed' });
    } finally {
      setIsTestingBrave(false);
    }
  };

  const handleTestOpenRouter = async () => {
    if (!openrouterApiKey && !settings.openrouter_api_key_set) {
      setOpenrouterTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingOpenRouter(true);
    setOpenrouterTestResult(null);
    try {
      // If input is empty but key is configured, pass null to test the saved key
      const keyToTest = openrouterApiKey || null;
      const result = await api.testOpenRouterKey(keyToTest);
      setOpenrouterTestResult(result);

      // Auto-save API key if validation succeeds and a new key was provided
      if (result.success && openrouterApiKey) {
        await api.updateSettings({ openrouter_api_key: openrouterApiKey });
        setOpenrouterApiKey(''); // Clear input after save

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setOpenrouterTestResult({ success: false, message: 'Test failed' });
    } finally {
      setIsTestingOpenRouter(false);
    }
  };

  const handleTestGroq = async () => {
    if (!groqApiKey && !settings.groq_api_key_set) {
      setGroqTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingGroq(true);
    setGroqTestResult(null);
    try {
      // If input is empty but key is configured, test with saved key via generic provider test
      // Note: backend/providers/groq.py must be registered with id 'groq'
      const result = await api.testProviderKey('groq', groqApiKey || 'saved');
      setGroqTestResult(result);

      // Auto-save API key if validation succeeds and a new key was provided
      if (result.success && groqApiKey) {
        await api.updateSettings({ groq_api_key: groqApiKey });
        setGroqApiKey(''); // Clear input after save

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setGroqTestResult({ success: false, message: 'Test failed' });
    } finally {
      setIsTestingGroq(false);
    }
  };

  const handleTestOllama = async () => {
    setIsTestingOllama(true);
    setOllamaTestResult(null);
    try {
      const result = await api.testOllamaConnection(ollamaBaseUrl);
      setOllamaTestResult(result);

      // Always refresh parent component's ollama status (success or failure)
      if (onRefreshOllama) {
        onRefreshOllama(ollamaBaseUrl);
      }

      if (result.success) {
        // Auto-save base URL if connection succeeds
        await api.updateSettings({ ollama_base_url: ollamaBaseUrl });

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setOllamaTestResult({ success: false, message: 'Connection failed' });

      // Refresh parent status on exception too
      if (onRefreshOllama) {
        onRefreshOllama(ollamaBaseUrl);
      }
    } finally {
      setIsTestingOllama(false);
    }
  };

  const handleCouncilModelChange = (index, modelId) => {
    setCouncilModels(prev => {
      const updated = [...prev];
      updated[index] = modelId;
      return updated;
    });
  };

  const getMemberFilter = (index) => {
    return councilMemberFilters[index] || 'remote';
  };

  const handleMemberFilterChange = (index, filter) => {
    setCouncilMemberFilters(prev => ({
      ...prev,
      [index]: filter
    }));

    // Clear the model selection for this member when switching filters
    setCouncilModels(prev => {
      const updated = [...prev];
      updated[index] = '';
      return updated;
    });
  };

  // Calculate Rate Limit Warning
  const getRateLimitWarning = () => {
    if (!settings || !availableModels || availableModels.length === 0) return null;

    let openRouterFreeCount = 0;
    let groqCount = 0; // Number of models (council, chairman, search) using Groq
    const totalCouncilMembers = councilModels.length;
    let totalRequestsPerRun = (totalCouncilMembers * 2) + 2; // Stage 1, Stage 2, Chairman, Search Query

    // Check OpenRouter free models
    councilModels.forEach(modelId => {
      const isRemote = !modelId.includes(':') || modelId.startsWith('openrouter:');
      if (isRemote) {
        const modelData = availableModels.find(m => m.id === modelId || m.id === modelId.replace('openrouter:', ''));
        if (modelData && modelData.is_free) {
          openRouterFreeCount++;
        }
      }
    });

    // Check Chairman and Search Query Model
    const chairmanModelData = availableModels.find(m => m.id === chairmanModel || m.id === chairmanModel.replace('openrouter:', ''));
    if (chairmanModelData && chairmanModelData.is_free && (!chairmanModel.includes(':') || chairmanModel.startsWith('openrouter:'))) {
      openRouterFreeCount++;
    }

    const searchQueryModelData = availableModels.find(m => m.id === searchQueryModel || m.id === searchQueryModel.replace('openrouter:', ''));
    if (searchQueryModelData && searchQueryModelData.is_free && (!searchQueryModel.includes(':') || searchQueryModel.startsWith('openrouter:'))) {
      openRouterFreeCount++;
    }

    // Logic for OpenRouter Warnings
    // OpenRouter: 20 RPM, 50 RPD (without credits)
    if (openRouterFreeCount > 0) {
      if (totalRequestsPerRun > 10 && openRouterFreeCount >= 3) { // 10 requests is approx half of 20 RPM
        return {
          type: 'error',
          title: 'High Rate Limit Risk (OpenRouter)',
          message: `Your council configuration generates ~${totalRequestsPerRun} requests per run, with ${openRouterFreeCount} free OpenRouter models. This may exceed the 20 requests/minute limit. Consider using Groq or Ollama for some members.`
        };
      } else if (openRouterFreeCount === totalRequestsPerRun) { // All requests from free OpenRouter
        return {
          type: 'warning',
          title: 'Daily Limit Caution (OpenRouter)',
          message: 'Free OpenRouter models are limited to 50 requests/day (without credits). Use Groq (14k/day) or Ollama for unlimited usage.'
        };
      }
    }

    // Logic for Groq Warnings
    // Groq: 30 RPM, 14,400 RPD (for Llama models)
    let groqRequests = 0;
    councilModels.forEach(id => {
      if (id.startsWith('groq:')) groqRequests += 2; // Stage 1 + Stage 2
    });
    if (chairmanModel.startsWith('groq:')) groqRequests += 1;
    if (searchQueryModel.startsWith('groq:')) groqRequests += 1;

    if (groqRequests > 15) {
      return {
        type: 'warning',
        title: 'High Concurrency Caution (Groq)',
        message: `Your configuration uses ${groqRequests} Groq requests per run. The free tier limit is 30 requests/minute. You may experience throttling if you send messages quickly.`
      };
    }

    return null;
  };

  const rateLimitWarning = getRateLimitWarning();

  const handleFeelingLucky = () => {
    // 1. Get pool of available models respecting "Free Only" filter
    let candidateModels = getFilteredAvailableModels();

    if (!candidateModels || candidateModels.length === 0) {
      setError("No models available to randomize! Check your enabled providers.");
      setTimeout(() => setError(null), 3000);
      return;
    }

    // Filter out models with known small context windows (< 8k) to prevent Stage 2 errors
    // Note: context_length might be undefined for some providers, we assume those are safe or unknown
    const safeModels = candidateModels.filter(m => !m.context_length || m.context_length >= 8192);

    // If we have enough safe models, use them. Otherwise fallback to all.
    if (safeModels.length >= 2) {
      candidateModels = safeModels;
    }

    // Helper to pick random item
    const pickRandom = (arr) => arr[Math.floor(Math.random() * arr.length)];

    // Helper to determine filter type (remote/local) from model ID
    const getFilterForModel = (modelId) => {
      return modelId.startsWith('ollama:') ? 'local' : 'remote';
    };

    // 2. Randomize Council Members (Unique if possible)
    let remainingModels = [...candidateModels];
    const newCouncilModels = [];
    const newMemberFilters = {};

    // We need to fill 'councilModels.length' slots
    for (let i = 0; i < councilModels.length; i++) {
      // If we ran out of unique models, refill the pool
      if (remainingModels.length === 0) {
        remainingModels = [...candidateModels];
      }

      const randomIndex = Math.floor(Math.random() * remainingModels.length);
      const selectedModel = remainingModels[randomIndex];

      newCouncilModels.push(selectedModel.id);
      newMemberFilters[i] = getFilterForModel(selectedModel.id);

      // Remove selected to avoid duplicates (until we run out)
      remainingModels.splice(randomIndex, 1);
    }

    // 3. Randomize Chairman
    const randomChairman = pickRandom(candidateModels);

    // 4. Randomize Search Query
    const randomSearch = pickRandom(candidateModels);

    // Apply Updates
    setCouncilModels(newCouncilModels);
    setCouncilMemberFilters(newMemberFilters);

    setChairmanModel(randomChairman.id);
    setChairmanFilter(getFilterForModel(randomChairman.id));

    setSearchQueryModel(randomSearch.id);
    setSearchQueryFilter(getFilterForModel(randomSearch.id));

    setSuccess(true);
    setTimeout(() => setSuccess(false), 2000);
  };

  const handleAddCouncilMember = () => {
    const newIndex = councilModels.length;

    // Determine best default filter based on what's available
    let defaultFilter = 'remote';
    const isRemoteAvailable = enabledProviders.openrouter || enabledProviders.direct || enabledProviders.groq;
    const isLocalAvailable = enabledProviders.ollama && ollamaAvailableModels.length > 0;

    if (!isRemoteAvailable && isLocalAvailable) {
      defaultFilter = 'local';
    }

    // Get models for the chosen filter
    const filtered = filterByRemoteLocal(getFilteredAvailableModels(), defaultFilter);

    // Even if no models found, we should allow adding the slot so user can switch filter/provider
    // But we try to pick a default if possible
    const defaultModel = filtered.length > 0 ? filtered[0].id : '';

    setCouncilModels(prev => [...prev, defaultModel]);

    // Initialize filter for new member
    setCouncilMemberFilters(prev => ({
      ...prev,
      [newIndex]: defaultFilter
    }));
  };

  const handleRemoveCouncilMember = (index) => {
    setCouncilModels(prev => prev.filter((_, i) => i !== index));
    // Clean up filters - shift indices down
    setCouncilMemberFilters(prev => {
      const newFilters = {};
      Object.keys(prev).forEach(key => {
        const idx = parseInt(key);
        if (idx < index) {
          newFilters[idx] = prev[idx];
        } else if (idx > index) {
          newFilters[idx - 1] = prev[idx];
        }
      });
      return newFilters;
    });
  };

  const handlePromptChange = (key, value) => {
    setPrompts(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleResetPrompt = async (key) => {
    try {
      const defaults = await api.getDefaultSettings();
      if (defaults[key]) {
        handlePromptChange(key, defaults[key]);
      }
    } catch (err) {
      console.error("Failed to fetch default prompt", err);
    }
  };

  const handleResetToDefaults = () => {
    setShowResetConfirm(true);
  };

  const confirmResetToDefaults = async () => {
    setShowResetConfirm(false);

    try {
      // 1. Disable all providers
      setEnabledProviders({
        openrouter: false,
        ollama: false,
        groq: false,
        direct: false
      });

      setDirectProviderToggles({
        openai: false,
        anthropic: false,
        google: false,
        mistral: false,
        deepseek: false
      });

      // 2. Reset Models to "Blank Slate" (User must select)
      // Initialize with 4 empty slots for council
      setCouncilModels(['', '', '', '']);
      setChairmanModel('');
      setSearchQueryModel('');

      // Reset filters to 'remote' default
      setCouncilMemberFilters({ 0: 'remote', 1: 'remote', 2: 'remote', 3: 'remote' });
      setChairmanFilter('remote');
      setSearchQueryFilter('remote');

      // 3. General Settings Defaults
      setSelectedSearchProvider('duckduckgo');
      setFullContentResults(3);
      setShowFreeOnly(false);
      setOllamaBaseUrl('http://localhost:11434');

      // 4. Reset Prompts to System Defaults (keep these useful)
      const defaults = await api.getDefaultSettings();
      setPrompts({
        stage1_prompt: defaults.stage1_prompt,
        stage2_prompt: defaults.stage2_prompt,
        stage3_prompt: defaults.stage3_prompt,
        search_query_prompt: defaults.search_query_prompt
      });

      // 5. Save the reset settings to backend
      const updates = {
        search_provider: 'duckduckgo',
        full_content_results: 3,
        enabled_providers: {
          openrouter: false,
          ollama: false,
          groq: false,
          direct: false
        },
        direct_provider_toggles: {
          openai: false,
          anthropic: false,
          google: false,
          mistral: false,
          deepseek: false
        },
        council_models: ['', '', '', ''],
        chairman_model: '',
        search_query_model: '',
        council_member_filters: { 0: 'remote', 1: 'remote', 2: 'remote', 3: 'remote' },
        chairman_filter: 'remote',
        search_query_filter: 'remote',
        stage1_prompt: defaults.stage1_prompt,
        stage2_prompt: defaults.stage2_prompt,
        stage3_prompt: defaults.stage3_prompt,
        search_query_prompt: defaults.search_query_prompt
      };
      await api.updateSettings(updates);

      setSuccess(true);
      // Navigate to Council Config so user sees the blank state
      setActiveSection('council');

      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError('Failed to reset settings');
    }
  };

  const handleTestDirectKey = async (providerId, keyField) => {
    const apiKey = directKeys[keyField];
    if (!apiKey) return;

    setValidatingKeys(prev => ({ ...prev, [providerId]: true }));
    setKeyValidationStatus(prev => ({ ...prev, [providerId]: null }));

    try {
      const result = await api.testProviderKey(providerId, apiKey);
      setKeyValidationStatus(prev => ({
        ...prev,
        [providerId]: {
          success: result.success,
          message: result.message
        }
      }));

      // Auto-save API key if validation succeeds
      if (result.success) {
        await api.updateSettings({ [keyField]: apiKey });
        setDirectKeys(prev => ({ ...prev, [keyField]: '' })); // Clear input after save

        // Reload settings
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setKeyValidationStatus(prev => ({
        ...prev,
        [providerId]: {
          success: false,
          message: err.message
        }
      }));
    } finally {
      setValidatingKeys(prev => ({ ...prev, [providerId]: false }));
    }
  };



  const handleExportCouncil = () => {
    const config = {
      // General
      search_provider: selectedSearchProvider,
      full_content_results: fullContentResults,
      show_free_only: showFreeOnly,

      // Enabled Providers
      enabled_providers: enabledProviders,

      // Council Configuration (unified)
      council_models: councilModels,
      chairman_model: chairmanModel,

      // Ollama Base URL
      ollama_base_url: ollamaBaseUrl,

      // Web Search Query Generator
      search_query_model: searchQueryModel,

      // Prompts
      prompts: prompts
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(config, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "council_config.json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleImportCouncil = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const config = JSON.parse(e.target.result);

        // Apply General Settings
        if (config.search_provider) setSelectedSearchProvider(config.search_provider);
        if (config.full_content_results !== undefined) setFullContentResults(config.full_content_results);
        if (config.show_free_only !== undefined) setShowFreeOnly(config.show_free_only);

        // Apply Enabled Providers
        if (config.enabled_providers) {
          setEnabledProviders(config.enabled_providers);
        }

        // Apply Council Configuration (unified)
        if (config.council_models) setCouncilModels(config.council_models);
        if (config.chairman_model) setChairmanModel(config.chairman_model);

        // Apply Ollama Base URL
        if (config.ollama_base_url) setOllamaBaseUrl(config.ollama_base_url);

        // Apply Web Search Query Generator
        if (config.search_query_model) setSearchQueryModel(config.search_query_model);

        // Apply Prompts
        if (config.prompts) {
          setPrompts(prev => ({ ...prev, ...config.prompts }));
        }

        // Validate imported models against all available models
        const allModels = getAllAvailableModels();
        const missingModels = (config.council_models || []).filter(id =>
          !allModels.find(m => m.id === id)
        );

        if (missingModels.length > 0) {
          setError(`Imported with warnings: Models not found: ${missingModels.join(', ')}`);
        } else {
          setSuccess(true);
          setTimeout(() => setSuccess(false), 3000);
        }

      } catch (err) {
        setError(`Import failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
    // Reset input
    event.target.value = '';
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    setSuccess(false);

    try {
      const updates = {
        search_provider: selectedSearchProvider,
        full_content_results: fullContentResults,

        // Enabled Providers
        enabled_providers: enabledProviders,
        direct_provider_toggles: directProviderToggles,

        // Council Configuration (unified)
        council_models: councilModels,
        chairman_model: chairmanModel,

        // Remote/Local filters for each selection
        council_member_filters: councilMemberFilters,
        chairman_filter: chairmanFilter,
        search_query_filter: searchQueryFilter,

        // Web Search Query Generator
        search_query_model: searchQueryModel,

        // Prompts
        ...prompts
      };

      // Only send API keys if they've been changed
      if (tavilyApiKey && !tavilyApiKey.startsWith('•')) {
        updates.tavily_api_key = tavilyApiKey;
      }
      if (braveApiKey && !braveApiKey.startsWith('•')) {
        updates.brave_api_key = braveApiKey;
      }
      if (openrouterApiKey && !openrouterApiKey.startsWith('•')) {
        updates.openrouter_api_key = openrouterApiKey;
      }
      if (groqApiKey && !groqApiKey.startsWith('•')) {
        updates.groq_api_key = groqApiKey;
      }

      // Add Direct Provider Keys
      Object.entries(directKeys).forEach(([key, value]) => {
        if (value && !value.startsWith('•')) {
          updates[key] = value;
        }
      });

      await api.updateSettings(updates);
      setSuccess(true);
      setTavilyApiKey('');
      setBraveApiKey('');
      setOpenrouterApiKey('');

      await loadSettings();
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError('Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  // Helper function to check if a direct provider is configured
  const isDirectProviderConfigured = (providerName) => {
    switch (providerName) {
      case 'OpenAI': return !!(directKeys.openai_api_key || settings?.openai_api_key_set);
      case 'Anthropic': return !!(directKeys.anthropic_api_key || settings?.anthropic_api_key_set);
      case 'Google': return !!(directKeys.google_api_key || settings?.google_api_key_set);
      case 'Mistral': return !!(directKeys.mistral_api_key || settings?.mistral_api_key_set);
      case 'DeepSeek': return !!(directKeys.deepseek_api_key || settings?.deepseek_api_key_set);
      default: return false;
    }
  };

  // Get all available models from all sources
  const getAllAvailableModels = () => {
    const models = [];

    // Add OpenRouter models if enabled
    if (enabledProviders.openrouter) {
      models.push(...availableModels);
    }

    // Add Ollama models if enabled
    if (enabledProviders.ollama) {
      models.push(...ollamaAvailableModels.map(m => ({
        ...m,
        id: `ollama:${m.id}`,
        name: `${m.name || m.id} (Local)`,
        provider: 'Ollama'
      })));
    }

    // Add Groq models if enabled
    if (enabledProviders.groq) {
      const groqModels = directAvailableModels.filter(m => m.provider === 'Groq');
      models.push(...groqModels);
    }

    // Add direct provider models if master toggle is enabled AND individual provider is enabled
    if (enabledProviders.direct) {
      const filteredDirectModels = directAvailableModels.filter(m => {
        if (m.provider === 'Groq') return false; // Handled separately above
        const providerKey = m.provider.toLowerCase();
        const individualToggleEnabled = directProviderToggles[providerKey];
        const providerConfigured = isDirectProviderConfigured(m.provider);
        return individualToggleEnabled && providerConfigured;
      });
      models.push(...filteredDirectModels);
    }

    // Deduplicate by model ID (prefer direct connections over OpenRouter for same model)
    // Since direct models are added last, always set to overwrite earlier entries
    const uniqueModels = new Map();
    models.forEach(model => {
      uniqueModels.set(model.id, model);
    });

    return Array.from(uniqueModels.values()).sort((a, b) => (a.name || '').localeCompare(b.name || ''));
  };

  // Get filtered models for council member selection (respects free filter)
  const getFilteredAvailableModels = () => {
    const all = getAllAvailableModels();
    if (!showFreeOnly) return all;

    // Filter to free models, but keep Ollama models (they're always "free")
    return all.filter(m => m.is_free || m.id.startsWith('ollama:'));
  };

  // Get models available for chairman (paid models only, unless it's Ollama)
  const getChairmanModels = () => {
    const all = getAllAvailableModels();
    // Chairman should be a paid/premium model (or local Ollama)
    return all.filter(m => !m.is_free || m.id.startsWith('ollama:'));
  };

  // Filter models by remote/local for specific use case
  const filterByRemoteLocal = (models, filter) => {
    if (filter === 'local') {
      // Only Ollama models
      return models.filter(m => m.id.startsWith('ollama:'));
    } else {
      // Remote: OpenRouter + Direct providers (exclude Ollama)
      return models.filter(m => !m.id.startsWith('ollama:'));
    }
  };

  if (!settings) {
    return (
      <div className="settings-overlay">
        <div className="settings-modal">
          <div className="settings-loading">Loading settings...</div>
        </div>
      </div>
    );
  }

  const selectedProviderInfo = SEARCH_PROVIDERS.find(p => p.id === selectedSearchProvider);

  const renderModelOptions = (models) => {
    // Group models by provider
    const grouped = models.reduce((acc, model) => {
      let providerLabel = model.provider; // Start with the provider field from backend

      if (model.provider === 'OpenRouter') {
        providerLabel = 'OpenRouter (Cloud)';
      } else if (model.provider === 'Ollama') {
        providerLabel = 'Local (Ollama)';
      } else {
        // For all other providers (OpenAI, Anthropic, Google, Mistral, DeepSeek, Groq)
        // from direct connections, append '(Direct)'
        providerLabel = `${model.provider} (Direct)`;
      }

      if (!acc[providerLabel]) acc[providerLabel] = [];
      acc[providerLabel].push(model);
      return acc;
    }, {});

    // Sort providers - prioritize direct, then specific vendors, then local
    const providerOrder = [
      'OpenAI (Direct)', 'Anthropic (Direct)', 'Google (Direct)', 'Mistral (Direct)', 'DeepSeek (Direct)',
      'Groq (Direct)',
      'OpenRouter (Cloud)',
      'Local (Ollama)'
    ];
    const sortedProviders = Object.keys(grouped).sort((a, b) => {
      const indexA = providerOrder.indexOf(a);
      const indexB = providerOrder.indexOf(b);
      if (indexA !== -1 && indexB !== -1) return indexA - indexB;
      if (indexA !== -1) return -1;
      if (indexB !== -1) return 1;
      return a.localeCompare(b);
    });

    return sortedProviders.map(provider => (
      <optgroup key={provider} label={provider}>
        {grouped[provider].map(model => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </optgroup>
    ));
  };

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-button" onClick={onClose}>&times;</button>
        </div>

        <div className="settings-body">
          {/* Sidebar Navigation */}
          <div className="settings-sidebar">
            <button
              className={`sidebar-nav-item ${activeSection === 'llm_keys' ? 'active' : ''}`}
              onClick={() => setActiveSection('llm_keys')}
            >
              LLM API Keys
            </button>
            <button
              className={`sidebar-nav-item ${activeSection === 'council' ? 'active' : ''}`}
              onClick={() => setActiveSection('council')}
            >
              Council Config
            </button>
            <button
              className={`sidebar-nav-item ${activeSection === 'prompts' ? 'active' : ''}`}
              onClick={() => setActiveSection('prompts')}
            >
              System Prompts
            </button>
            <button
              className={`sidebar-nav-item ${activeSection === 'search' ? 'active' : ''}`}
              onClick={() => setActiveSection('search')}
            >
              Search Providers
            </button>
            <button
              className={`sidebar-nav-item ${activeSection === 'import_export' ? 'active' : ''}`}
              onClick={() => setActiveSection('import_export')}
            >
              Backup & Reset
            </button>
          </div>

          {/* Main Content Area */}
          <div className="settings-main-panel">

            {/* API KEYS (LLM API Keys) */}
            {activeSection === 'llm_keys' && (
              <section className="settings-section">
                <h3>API Credentials</h3>
                <p className="section-description">
                  Configure keys for LLM providers.
                  Keys are <strong>auto-saved</strong> immediately upon successful test.
                </p>

                {/* OpenRouter */}
                <div className="api-key-section">
                  <label>OpenRouter API Key</label>
                  <div className="api-key-input-row">
                    <input
                      type="password"
                      placeholder={settings?.openrouter_api_key_set ? '••••••••••••••••' : 'Enter API key'}
                      value={openrouterApiKey}
                      onChange={(e) => {
                        setOpenrouterApiKey(e.target.value);
                        setOpenrouterTestResult(null);
                      }}
                      className={settings?.openrouter_api_key_set && !openrouterApiKey ? 'key-configured' : ''}
                    />
                    <button
                      className="test-button"
                      onClick={handleTestOpenRouter}
                      disabled={!openrouterApiKey && !settings?.openrouter_api_key_set || isTestingOpenRouter}
                    >
                      {isTestingOpenRouter ? 'Testing...' : (settings?.openrouter_api_key_set && !openrouterApiKey ? 'Retest' : 'Test')}
                    </button>
                  </div>
                  {settings?.openrouter_api_key_set && !openrouterApiKey && (
                    <div className="key-status set">✓ API key configured</div>
                  )}
                  {openrouterTestResult && (
                    <div className={`test-result ${openrouterTestResult.success ? 'success' : 'error'}`}>
                      {openrouterTestResult.message}
                    </div>
                  )}
                  <p className="api-key-hint">
                    Get key at <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer">openrouter.ai</a>
                  </p>
                </div>

                {/* Groq */}
                <div className="api-key-section">
                  <label>Groq API Key</label>
                  <div className="api-key-input-row">
                    <input
                      type="password"
                      placeholder={settings?.groq_api_key_set ? '••••••••••••••••' : 'Enter API key'}
                      value={groqApiKey}
                      onChange={(e) => {
                        setGroqApiKey(e.target.value);
                        setGroqTestResult(null);
                      }}
                      className={settings?.groq_api_key_set && !groqApiKey ? 'key-configured' : ''}
                    />
                    <button
                      className="test-button"
                      onClick={handleTestGroq}
                      disabled={!groqApiKey && !settings?.groq_api_key_set || isTestingGroq}
                    >
                      {isTestingGroq ? 'Testing...' : (settings?.groq_api_key_set && !groqApiKey ? 'Retest' : 'Test')}
                    </button>
                  </div>
                  {settings?.groq_api_key_set && !groqApiKey && (
                    <div className="key-status set">✓ API key configured</div>
                  )}
                  {groqTestResult && (
                    <div className={`test-result ${groqTestResult.success ? 'success' : 'error'}`}>
                      {groqTestResult.message}
                    </div>
                  )}
                  <p className="api-key-hint">
                    Get key at <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">console.groq.com</a>
                  </p>
                </div>

                {/* Ollama */}
                <div className="api-key-section">
                  <label>Ollama Base URL</label>
                  <div className="api-key-input-row">
                    <input
                      type="text"
                      placeholder="http://localhost:11434"
                      value={ollamaBaseUrl}
                      onChange={(e) => {
                        setOllamaBaseUrl(e.target.value);
                        setOllamaTestResult(null);
                      }}
                    />
                    <button
                      className="test-button"
                      onClick={handleTestOllama}
                      disabled={!ollamaBaseUrl || isTestingOllama}
                    >
                      {isTestingOllama ? 'Testing...' : 'Connect'}
                    </button>
                  </div>
                  {ollamaTestResult && (
                    <div className={`test-result ${ollamaTestResult.success ? 'success' : 'error'}`}>
                      {ollamaTestResult.message}
                    </div>
                  )}
                  {ollamaStatus && ollamaStatus.connected && (
                    <div className="ollama-auto-status connected">
                      <span className="status-indicator connected">●</span>
                      <span className="status-text">
                        <strong>Connected</strong> <span className="status-separator">·</span> <span className="status-time">Last: {new Date(ollamaStatus.lastConnected).toLocaleTimeString()}</span>
                      </span>
                    </div>
                  )}
                  {ollamaStatus && !ollamaStatus.connected && !ollamaStatus.testing && (
                    <div className="ollama-auto-status">
                      <span className="status-indicator disconnected">●</span>
                      <span className="status-text">Not connected</span>
                    </div>
                  )}
                  <div className="model-options-row" style={{ marginTop: '12px' }}>
                    <button
                      type="button"
                      className="reset-defaults-button"
                      onClick={() => loadOllamaModels(ollamaBaseUrl)}
                    >
                      Refresh Local Models
                    </button>
                  </div>
                </div>

                {/* Direct LLM API Connections */}
                <div className="subsection" style={{ marginTop: '24px' }}>
                  <h4>Direct LLM Connections</h4>
                  {DIRECT_PROVIDERS.map(dp => (
                    <div key={dp.id} className="api-key-section">
                      <label>{dp.name} API Key</label>
                      <div className="api-key-input-row">
                        <input
                          type="password"
                          placeholder={settings?.[`${dp.key}_set`] ? '••••••••••••••••' : 'Enter API key'}
                          value={directKeys[dp.key]}
                          onChange={e => setDirectKeys(prev => ({ ...prev, [dp.key]: e.target.value }))}
                          className={settings?.[`${dp.key}_set`] && !directKeys[dp.key] ? 'key-configured' : ''}
                        />
                        <button
                          className="test-button"
                          onClick={() => handleTestDirectKey(dp.id, dp.key)}
                          disabled={(!directKeys[dp.key] && !settings?.[`${dp.key}_set`]) || validatingKeys[dp.id]}
                        >
                          {validatingKeys[dp.id] ? 'Testing...' : (settings?.[`${dp.key}_set`] && !directKeys[dp.key] ? 'Retest' : 'Test')}
                        </button>
                      </div>
                      {settings?.[`${dp.key}_set`] && !directKeys[dp.key] && (
                        <div className="key-status set">✓ API key configured</div>
                      )}
                      {keyValidationStatus[dp.id] && (
                        <div className={`test-result ${keyValidationStatus[dp.id].success ? 'success' : 'error'}`}>
                          {keyValidationStatus[dp.id].message}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* COUNCIL CONFIGURATION */}
            {activeSection === 'council' && (
              <>
                <section className="settings-section">
                  <h3>Available Model Sources</h3>
                  <p className="section-description">
                    Toggle which providers are available for the search generator, council members, and chairman.
                    <br /><em style={{ opacity: 0.7, fontSize: '12px' }}>Note: Non-chat models (embeddings, image generation, speech, OCR, etc.) are automatically filtered out.</em>
                  </p>

                  <div className="hybrid-settings-card">
                    {/* Primary Sources */}
                    <div className="filter-group">
                      <label className="toggle-wrapper">
                        <div className="toggle-switch">
                          <input
                            type="checkbox"
                            checked={enabledProviders.openrouter}
                            onChange={(e) => setEnabledProviders(prev => ({ ...prev, openrouter: e.target.checked }))}
                          />
                          <span className="slider"></span>
                        </div>
                        <span className="toggle-text">OpenRouter (Cloud)</span>
                      </label>

                      <label className="toggle-wrapper">
                        <div className="toggle-switch">
                          <input
                            type="checkbox"
                            checked={enabledProviders.ollama}
                            onChange={(e) => setEnabledProviders(prev => ({ ...prev, ollama: e.target.checked }))}
                          />
                          <span className="slider"></span>
                        </div>
                        <span className="toggle-text">Local (Ollama)</span>
                      </label>

                      <label className="toggle-wrapper">
                        <div className="toggle-switch">
                          <input
                            type="checkbox"
                            checked={enabledProviders.groq}
                            onChange={(e) => setEnabledProviders(prev => ({ ...prev, groq: e.target.checked }))}
                          />
                          <span className="slider"></span>
                        </div>
                        <span className="toggle-text">Groq (Fast Inference)</span>
                      </label>
                    </div>

                    <div className="filter-divider"></div>

                    {/* Direct Connections Master Toggle */}
                    <div className="filter-group" style={{ marginBottom: '12px' }}>
                      <label className="toggle-wrapper">
                        <div className="toggle-switch">
                          <input
                            type="checkbox"
                            checked={enabledProviders.direct}
                            onChange={(e) => {
                              const isEnabled = e.target.checked;
                              setEnabledProviders(prev => ({ ...prev, direct: isEnabled }));
                              // If master turned off, disable all children
                              if (!isEnabled) {
                                setDirectProviderToggles({
                                  openai: false,
                                  anthropic: false,
                                  google: false,
                                  mistral: false,
                                  deepseek: false
                                });
                              }
                            }}
                          />
                          <span className="slider"></span>
                        </div>
                        <span className="toggle-text">Direct Connections</span>
                      </label>
                    </div>

                    {/* Individual Direct Provider Toggles (purple) */}
                    <div className="direct-grid" style={{ opacity: enabledProviders.direct ? 1 : 0.7 }}>
                      {DIRECT_PROVIDERS.map(dp => (
                        <label key={dp.id} className="toggle-wrapper">
                          <div className="toggle-switch direct-toggle">
                            <input
                              type="checkbox"
                              checked={directProviderToggles[dp.id]}
                              onChange={(e) => {
                                const isEnabled = e.target.checked;
                                setDirectProviderToggles(prev => {
                                  const newState = { ...prev, [dp.id]: isEnabled };

                                  // Auto-enable master if any child is enabled
                                  if (isEnabled && !enabledProviders.direct) {
                                    setEnabledProviders(prevEP => ({ ...prevEP, direct: true }));
                                  }

                                  // Auto-disable master if ALL children are disabled
                                  const hasAnyEnabled = Object.values(newState).some(v => v);
                                  if (!hasAnyEnabled && enabledProviders.direct) {
                                    setEnabledProviders(prevEP => ({ ...prevEP, direct: false }));
                                  }

                                  return newState;
                                });
                              }}
                            />
                            <span className="slider"></span>
                          </div>
                          <span className="toggle-text" style={{ fontSize: '13px' }}>
                            {dp.name}
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                </section>

                <section className="settings-section">
                  <h3>Council Composition</h3>
                  <div className="model-options-row">
                    <div className="model-filter-controls">
                      <label className="free-filter-label" style={{ opacity: enabledProviders.openrouter ? 1 : 0.3, cursor: enabledProviders.openrouter ? 'pointer' : 'not-allowed' }}>
                        <input
                          type="checkbox"
                          checked={showFreeOnly}
                          onChange={e => setShowFreeOnly(e.target.checked)}
                          disabled={!enabledProviders.openrouter}
                        />
                        Show free OpenRouter models only
                        <div className="info-tooltip-container">
                          <span className="info-icon">i</span>
                          <div className="info-tooltip">
                            Free OpenRouter models are limited to 20 requests/minute and 50/day (without credits). Large councils generate many requests at once.
                          </div>
                        </div>
                      </label>
                      {isLoadingModels && <span className="loading-models">Loading models...</span>}
                    </div>
                  </div>
                  <div className="lucky-button-container">
                    <button
                      type="button"
                      className="lucky-button"
                      onClick={handleFeelingLucky}
                      title="Randomize models from enabled sources"
                    >
                      🎲 I'm Feeling Lucky
                    </button>
                  </div>
                  {/* Council Members */}                                  <div className="subsection" style={{ marginTop: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h4 style={{ margin: 0 }}>Council Members</h4>
                    </div>
                    <div className="council-members">
                      {councilModels.map((modelId, index) => {
                        const memberFilter = getMemberFilter(index);
                        return (
                          <div key={index} className="council-member-row">
                            <span className="member-label">Member {index + 1}</span>
                            <div className="model-type-toggle">
                              <button
                                type="button"
                                className={`type-btn ${memberFilter === 'remote' ? 'active' : ''}`}
                                onClick={() => handleMemberFilterChange(index, 'remote')}
                                disabled={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq}
                                title={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq ? 'Enable OpenRouter, Groq, or Direct Connections first' : ''}
                              >
                                Remote
                              </button>
                              <button
                                type="button"
                                className={`type-btn ${memberFilter === 'local' ? 'active' : ''}`}
                                onClick={() => handleMemberFilterChange(index, 'local')}
                                disabled={!enabledProviders.ollama || ollamaAvailableModels.length === 0}
                                title={!enabledProviders.ollama || ollamaAvailableModels.length === 0 ? 'Enable and connect Ollama first' : ''}
                              >
                                Local
                              </button>
                            </div>
                            <select
                              value={modelId}
                              onChange={e => handleCouncilModelChange(index, e.target.value)}
                              className="model-select"
                              disabled={isLoadingModels && availableModels.length === 0}
                            >
                              <option value="">{isLoadingModels && availableModels.length === 0 ? "Loading models..." : "Select a model"}</option>
                              {renderModelOptions(filterByRemoteLocal(getFilteredAvailableModels(), memberFilter))}
                              {/* Keep current selection visible even if filtered out */}
                              {!filterByRemoteLocal(getFilteredAvailableModels(), memberFilter).find(m => m.id === modelId) && modelId && (
                                <option value={modelId}>
                                  {isLoadingModels ? "Loading model info..." : (getAllAvailableModels().find(m => m.id === modelId)?.name || modelId)}
                                </option>
                              )}
                            </select>
                            {index >= 2 && (
                              <button
                                type="button"
                                className="remove-member-button"
                                onClick={() => handleRemoveCouncilMember(index)}
                                title="Remove member"
                              >
                                ×
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    <button
                      type="button"
                      className="add-member-button"
                      onClick={handleAddCouncilMember}
                      disabled={getFilteredAvailableModels().length === 0 || councilModels.length >= 8}
                    >
                      + Add Council Member
                    </button>
                    <p className="section-description" style={{ marginTop: '8px', marginBottom: '0' }}>
                      Max 8 members. With 6+ members, requests are processed in batches.
                    </p>
                    {councilModels.length >= 6 && (
                      <div className="council-size-warning">
                        ⚠️ <strong>6+ members:</strong> Requests will be processed in batches of 3 to avoid rate limits.
                      </div>
                    )}

                    {/* Rate Limit Warning Banner */}
                    {rateLimitWarning && (
                      <div className={`rate-limit-warning ${rateLimitWarning.type}`}>
                        <span className="warning-icon">
                          {rateLimitWarning.type === 'error' ? '🛑' : '⚠️'}
                        </span>
                        <div>
                          <strong>{rateLimitWarning.title}</strong><br />
                          {rateLimitWarning.message}
                        </div>
                      </div>
                    )}
                  </div>
                  {/* Chairman */}
                  <div className="subsection" style={{ marginTop: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <h4 style={{ margin: 0 }}>Chairman Model</h4>
                      <div className="model-type-toggle">
                        <button
                          type="button"
                          className={`type-btn ${chairmanFilter === 'remote' ? 'active' : ''}`}
                          onClick={() => {
                            setChairmanFilter('remote');
                            setChairmanModel('');
                          }}
                          disabled={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq}
                          title={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq ? 'Enable OpenRouter, Groq, or Direct Connections first' : ''}
                        >
                          Remote
                        </button>
                        <button
                          type="button"
                          className={`type-btn ${chairmanFilter === 'local' ? 'active' : ''}`}
                          onClick={() => {
                            setChairmanFilter('local');
                            setChairmanModel('');
                          }}
                          disabled={!enabledProviders.ollama || ollamaAvailableModels.length === 0}
                          title={!enabledProviders.ollama || ollamaAvailableModels.length === 0 ? 'Enable and connect Ollama first' : ''}
                        >
                          Local
                        </button>
                      </div>
                    </div>
                    <div className="chairman-selection">
                      <select
                        value={chairmanModel}
                        onChange={(e) => setChairmanModel(e.target.value)}
                        className="model-select"
                      >
                        <option value="">Select a model</option>
                        {renderModelOptions(filterByRemoteLocal(getFilteredAvailableModels(), chairmanFilter))}
                      </select>
                    </div>
                  </div>

                  {/* Web Search Query Generator */}
                  <div className="subsection" style={{ marginTop: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <h4 style={{ margin: 0 }}>Search Query Generator</h4>
                      <div className="model-type-toggle">
                        <button
                          type="button"
                          className={`type-btn ${searchQueryFilter === 'remote' ? 'active' : ''}`}
                          onClick={() => {
                            setSearchQueryFilter('remote');
                            setSearchQueryModel('');
                          }}
                          disabled={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq}
                          title={!enabledProviders.openrouter && !enabledProviders.direct && !enabledProviders.groq ? 'Enable OpenRouter, Groq, or Direct Connections first' : ''}
                        >
                          Remote
                        </button>
                        <button
                          type="button"
                          className={`type-btn ${searchQueryFilter === 'local' ? 'active' : ''}`}
                          onClick={() => {
                            setSearchQueryFilter('local');
                            setSearchQueryModel('');
                          }}
                          disabled={!enabledProviders.ollama || ollamaAvailableModels.length === 0}
                          title={!enabledProviders.ollama || ollamaAvailableModels.length === 0 ? 'Enable and connect Ollama first' : ''}
                        >
                          Local
                        </button>
                      </div>
                    </div>
                    <p className="section-description" style={{ marginBottom: '8px' }}>
                      Generates optimized search terms from user questions.
                    </p>
                    <div className="chairman-selection">
                      <select
                        value={searchQueryModel}
                        onChange={(e) => setSearchQueryModel(e.target.value)}
                        className="model-select"
                      >
                        <option value="">Select a model</option>
                        {renderModelOptions(filterByRemoteLocal(getAllAvailableModels(), searchQueryFilter))}
                      </select>
                    </div>
                  </div>
                </section>
              </>
            )}

            {/* SYSTEM PROMPTS */}
            {activeSection === 'prompts' && (
              <section className="settings-section">
                <h3>System Prompts</h3>
                <p className="section-description">
                  Customize the instructions given to the models at each stage.
                </p>

                <div className="prompts-tabs">
                  <button
                    className={`prompt-tab ${activePromptTab === 'stage1' ? 'active' : ''}`}
                    onClick={() => setActivePromptTab('stage1')}
                  >
                    Stage 1
                  </button>
                  <button
                    className={`prompt-tab ${activePromptTab === 'stage2' ? 'active' : ''}`}
                    onClick={() => setActivePromptTab('stage2')}
                  >
                    Stage 2
                  </button>
                  <button
                    className={`prompt-tab ${activePromptTab === 'stage3' ? 'active' : ''}`}
                    onClick={() => setActivePromptTab('stage3')}
                  >
                    Stage 3
                  </button>
                  <button
                    className={`prompt-tab ${activePromptTab === 'search' ? 'active' : ''}`}
                    onClick={() => setActivePromptTab('search')}
                  >
                    Search Query
                  </button>
                </div>

                <div className="prompt-editor">
                  {activePromptTab === 'search' && (
                    <div className="prompt-content">
                      <label>Search Query Generation</label>
                      <p className="section-description" style={{ marginBottom: '10px' }}>
                        Generates optimized search terms from user questions for web search.
                      </p>
                      <p className="prompt-help">Variables: <code>{'{user_query}'}</code></p>
                      <textarea
                        value={prompts.search_query_prompt}
                        onChange={(e) => handlePromptChange('search_query_prompt', e.target.value)}
                        rows={10}
                      />
                      <button className="reset-prompt-btn" onClick={() => handleResetPrompt('search_query_prompt')}>Reset to Default</button>
                    </div>
                  )}
                  {activePromptTab === 'stage1' && (
                    <div className="prompt-content">
                      <label>Stage 1: Initial Response</label>
                      <p className="section-description" style={{ marginBottom: '10px' }}>
                        Guides council members' initial responses to user questions.
                      </p>
                      <p className="prompt-help">Variables: <code>{'{user_query}'}</code>, <code>{'{search_context_block}'}</code></p>
                      <textarea
                        value={prompts.stage1_prompt}
                        onChange={(e) => handlePromptChange('stage1_prompt', e.target.value)}
                        rows={15}
                      />
                      <button className="reset-prompt-btn" onClick={() => handleResetPrompt('stage1_prompt')}>Reset to Default</button>
                    </div>
                  )}
                  {activePromptTab === 'stage2' && (
                    <div className="prompt-content">
                      <label>Stage 2: Peer Ranking</label>
                      <p className="section-description" style={{ marginBottom: '10px' }}>
                        Instructs models how to rank and evaluate peer responses.
                      </p>
                      <p className="prompt-help">Variables: <code>{'{user_query}'}</code>, <code>{'{responses_text}'}</code>, <code>{'{search_context_block}'}</code></p>
                      <textarea
                        value={prompts.stage2_prompt}
                        onChange={(e) => handlePromptChange('stage2_prompt', e.target.value)}
                        rows={15}
                      />
                      <button className="reset-prompt-btn" onClick={() => handleResetPrompt('stage2_prompt')}>Reset to Default</button>
                    </div>
                  )}
                  {activePromptTab === 'stage3' && (
                    <div className="prompt-content">
                      <label>Stage 3: Chairman Synthesis</label>
                      <p className="section-description" style={{ marginBottom: '10px' }}>
                        Directs the chairman to synthesize a final answer from all inputs.
                      </p>
                      <p className="prompt-help">Variables: <code>{'{user_query}'}</code>, <code>{'{stage1_text}'}</code>, <code>{'{stage2_text}'}</code>, <code>{'{search_context_block}'}</code></p>
                      <textarea
                        value={prompts.stage3_prompt}
                        onChange={(e) => handlePromptChange('stage3_prompt', e.target.value)}
                        rows={15}
                      />
                      <button className="reset-prompt-btn" onClick={() => handleResetPrompt('stage3_prompt')}>Reset to Default</button>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* SEARCH PROVIDERS (New Section) */}
            {activeSection === 'search' && (
              <section className="settings-section">
                <h3>Web Search Provider</h3>
                <div className="provider-options">
                  {SEARCH_PROVIDERS.map(provider => (
                    <div key={provider.id} className={`provider-option-container ${selectedSearchProvider === provider.id ? 'selected' : ''}`}>
                      <label className="provider-option">
                        <input
                          type="radio"
                          name="search_provider"
                          value={provider.id}
                          checked={selectedSearchProvider === provider.id}
                          onChange={() => setSelectedSearchProvider(provider.id)}
                        />
                        <div className="provider-info">
                          <span className="provider-name">{provider.name}</span>
                          <span className="provider-description">{provider.description}</span>
                        </div>
                      </label>

                      {/* Inline API Key Input for Tavily */}
                      {selectedSearchProvider === 'tavily' && provider.id === 'tavily' && (
                        <div className="inline-api-key-section">
                          <div className="api-key-input-row">
                            <input
                              type="password"
                              placeholder={settings.tavily_api_key_set ? '••••••••••••••••' : 'Enter Tavily API key'}
                              value={tavilyApiKey}
                              onChange={e => {
                                setTavilyApiKey(e.target.value);
                                setTavilyTestResult(null);
                              }}
                              className={settings.tavily_api_key_set && !tavilyApiKey ? 'key-configured' : ''}
                            />
                            <button
                              type="button"
                              className="test-button"
                              onClick={handleTestTavily}
                              disabled={isTestingTavily || (!tavilyApiKey && !settings.tavily_api_key_set)}
                            >
                              {isTestingTavily ? 'Testing...' : (settings.tavily_api_key_set && !tavilyApiKey ? 'Retest' : 'Test')}
                            </button>
                          </div>
                          {settings.tavily_api_key_set && !tavilyApiKey && (
                            <div className="key-status set">✓ API key configured</div>
                          )}
                          {tavilyTestResult && (
                            <div className={`test-result ${tavilyTestResult.success ? 'success' : 'error'}`}>
                              {tavilyTestResult.success ? '✓' : '✗'} {tavilyTestResult.message}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Inline API Key Input for Brave */}
                      {selectedSearchProvider === 'brave' && provider.id === 'brave' && (
                        <div className="inline-api-key-section">
                          <div className="api-key-input-row">
                            <input
                              type="password"
                              placeholder={settings.brave_api_key_set ? '••••••••••••••••' : 'Enter Brave API key'}
                              value={braveApiKey}
                              onChange={e => {
                                setBraveApiKey(e.target.value);
                                setBraveTestResult(null);
                              }}
                              className={settings.brave_api_key_set && !braveApiKey ? 'key-configured' : ''}
                            />
                            <button
                              type="button"
                              className="test-button"
                              onClick={handleTestBrave}
                              disabled={isTestingBrave || (!braveApiKey && !settings.brave_api_key_set)}
                            >
                              {isTestingBrave ? 'Testing...' : (settings.brave_api_key_set && !braveApiKey ? 'Retest' : 'Test')}
                            </button>
                          </div>
                          {settings.brave_api_key_set && !braveApiKey && (
                            <div className="key-status set">✓ API key configured</div>
                          )}
                          {braveTestResult && (
                            <div className={`test-result ${braveTestResult.success ? 'success' : 'error'}`}>
                              {braveTestResult.success ? '✓' : '✗'} {braveTestResult.message}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className="full-content-section">
                  <label>Full Article Fetch (Jina AI)</label>
                  <p className="setting-description">
                    Uses Jina AI to read the full text of the top search results.
                    <strong> Set to 0 to disable.</strong>
                  </p>
                  <div className="full-content-input-row">
                    <input
                      type="range"
                      min="0"
                      max="5"
                      value={fullContentResults}
                      onChange={e => setFullContentResults(parseInt(e.target.value, 10))}
                      className="full-content-slider"
                    />
                    <span className="full-content-value">{fullContentResults} results</span>
                  </div>
                </div>
              </section>
            )}

            {/* IMPORT & EXPORT (New Section) */}
            {activeSection === 'import_export' && (
              <section className="settings-section">
                <h3>Backup & Reset</h3>
                <p className="section-description">
                  Save or restore your council configuration (models, prompts, settings).
                  <br /><em>Note: API keys are NOT exported for security.</em>
                </p>

                <div className="subsection">
                  <div className="council-actions" style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <input
                      type="file"
                      id="import-council"
                      style={{ display: 'none' }}
                      accept=".json"
                      onChange={handleImportCouncil}
                    />
                    <button
                      className="action-btn"
                      onClick={() => document.getElementById('import-council').click()}
                      title="Import Configuration"
                    >
                      Import Config
                    </button>
                    <button
                      className="action-btn"
                      onClick={handleExportCouncil}
                      title="Export Configuration"
                    >
                      Export Config
                    </button>
                  </div>
                </div>

                <div className="subsection" style={{ marginTop: '32px', paddingTop: '20px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <h4 style={{ color: '#f87171' }}>Danger Zone</h4>
                  <p className="section-description">
                    Reset all settings to their default values. This will clear your council selection and custom prompts.
                    API keys will be preserved.
                  </p>
                  <button
                    className="reset-button"
                    type="button"
                    onClick={handleResetToDefaults}
                    style={{ marginTop: '10px' }}
                  >
                    Reset to Defaults
                  </button>
                </div>
              </section>
            )}

          </div>
        </div>

        <div className="settings-footer">
          {error && <div className="settings-error">{error}</div>}
          {success && (
            <div className="settings-success">
              {activeSection === 'llm_keys' && !settings?.openrouter_api_key_set && !ollamaStatus?.connected
                ? 'Defaults loaded. Please configure an API Key.'
                : 'Settings saved!'}
            </div>
          )}

          <div className="footer-actions">
            <button className="cancel-button" onClick={onClose}>
              Close
            </button>
            <button
              className="save-button"
              onClick={handleSave}
              disabled={isSaving || !hasChanges}
            >
              {isSaving ? 'Saving...' : (success ? 'Saved!' : 'Save Changes')}
            </button>
          </div>
        </div>
      </div>

      {showResetConfirm && (
        <div className="settings-overlay confirmation-overlay" onClick={() => setShowResetConfirm(false)}>
          <div className="settings-modal confirmation-modal" onClick={e => e.stopPropagation()}>
            <div className="settings-header">
              <h2>Confirm Reset</h2>
            </div>
            <div className="settings-content confirmation-content">
              <p>Are you sure you want to reset to defaults?</p>
              <div className="confirmation-details">
                <p><strong>This will reset:</strong></p>
                <ul>
                  <li>All model selections</li>
                  <li>System prompts</li>
                  <li>General settings</li>
                </ul>
                <p className="confirmation-safe">✓ API keys will be PRESERVED</p>
              </div>
            </div>
            <div className="settings-footer">
              <div className="footer-actions" style={{ width: '100%', justifyContent: 'flex-end' }}>
                <button className="cancel-button" onClick={() => setShowResetConfirm(false)}>Cancel</button>
                <button className="reset-button" onClick={confirmResetToDefaults}>Confirm Reset</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}