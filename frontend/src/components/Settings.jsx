import { useState, useEffect, useMemo, useRef } from 'react';
import { api } from '../api';
import SearchableModelSelect from './SearchableModelSelect';
import ProviderSettings from './settings/ProviderSettings';
import CouncilConfig from './settings/CouncilConfig';
import SearchSettings from './settings/SearchSettings';
import PromptSettings from './settings/PromptSettings';
import './Settings.css';





export default function Settings({ onClose, ollamaStatus, onRefreshOllama, initialSection = 'llm_keys' }) {
  const [activeSection, setActiveSection] = useState(initialSection); // 'llm_keys', 'council', 'prompts', 'search', 'import_export'

  const [settings, setSettings] = useState(null);
  const [selectedSearchProvider, setSelectedSearchProvider] = useState('duckduckgo');
  const [searchKeywordExtraction, setSearchKeywordExtraction] = useState('direct');
  const [fullContentResults, setFullContentResults] = useState(3);
  const [searchResultCount, setSearchResultCount] = useState(8);
  const [searchHybridMode, setSearchHybridMode] = useState(true);

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

  // Custom OpenAI-compatible Endpoint State
  const [customEndpointName, setCustomEndpointName] = useState('');
  const [customEndpointUrl, setCustomEndpointUrl] = useState('');
  const [customEndpointApiKey, setCustomEndpointApiKey] = useState('');
  const [customEndpointModels, setCustomEndpointModels] = useState([]);
  const [isTestingCustomEndpoint, setIsTestingCustomEndpoint] = useState(false);
  const [customEndpointTestResult, setCustomEndpointTestResult] = useState(null);

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
  const [serperApiKey, setSerperApiKey] = useState('');
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [braveApiKey, setBraveApiKey] = useState('');
  const [isTestingSerper, setIsTestingSerper] = useState(false);
  const [isTestingTavily, setIsTestingTavily] = useState(false);
  const [isTestingBrave, setIsTestingBrave] = useState(false);
  const [serperTestResult, setSerperTestResult] = useState(null);
  const [tavilyTestResult, setTavilyTestResult] = useState(null);
  const [braveTestResult, setBraveTestResult] = useState(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Enabled Providers (which sources are available)
  const [enabledProviders, setEnabledProviders] = useState({
    openrouter: true,
    ollama: false,
    groq: false,
    direct: false,  // Master toggle for all direct connections
    custom: false   // Custom OpenAI-compatible endpoint
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
  const [councilTemperature, setCouncilTemperature] = useState(0.5);
  const [chairmanTemperature, setChairmanTemperature] = useState(0.4);
  const [stage2Temperature, setStage2Temperature] = useState(0.3);

  // System Prompts State
  const [prompts, setPrompts] = useState({
    stage1_prompt: '',
    stage2_prompt: '',
    stage3_prompt: '',
    title_prompt: '',

  });
  const [activePromptTab, setActivePromptTab] = useState('stage1');

  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showFreeOnly, setShowFreeOnly] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});

  // Ref for chairman select to focus on validation error
  const chairmanSelectRef = useRef(null);

  // Remote/Local filter toggles per model type
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
      if (searchKeywordExtraction !== (settings.search_keyword_extraction || 'direct')) return true;
      if (fullContentResults !== (settings.full_content_results ?? 3)) return true;
      if (searchResultCount !== (settings.search_result_count ?? 8)) return true;
      if (searchHybridMode !== (settings.search_hybrid_mode ?? true)) return true;
      if (showFreeOnly !== (settings.show_free_only ?? false)) return true;

      // Enabled Providers
      if (JSON.stringify(enabledProviders) !== JSON.stringify(settings.enabled_providers)) return true;
      if (JSON.stringify(directProviderToggles) !== JSON.stringify(settings.direct_provider_toggles)) return true;

      // Council Configuration (unified)
      if (JSON.stringify(councilModels) !== JSON.stringify(settings.council_models)) return true;
      if (chairmanModel !== settings.chairman_model) return true;
      if (councilTemperature !== (settings.council_temperature ?? 0.5)) return true;
      if (chairmanTemperature !== (settings.chairman_temperature ?? 0.4)) return true;
      if (stage2Temperature !== (settings.stage2_temperature ?? 0.3)) return true;

      // Remote/Local filters
      if (JSON.stringify(councilMemberFilters) !== JSON.stringify(settings.council_member_filters || {})) return true;
      if (chairmanFilter !== (settings.chairman_filter || 'remote')) return true;
      // Prompts
      if (prompts.stage1_prompt !== settings.stage1_prompt) return true;
      if (prompts.stage2_prompt !== settings.stage2_prompt) return true;
      if (prompts.stage3_prompt !== settings.stage3_prompt) return true;

      // Note: API keys are auto-saved on test, so we don't check them here

      return false;
    };

    setHasChanges(checkChanges());
  }, [
    settings,
    selectedSearchProvider,
    searchKeywordExtraction,
    fullContentResults,
    searchResultCount,
    searchHybridMode,
    showFreeOnly,
    enabledProviders,
    directProviderToggles,
    councilModels,
    chairmanModel,
    councilTemperature,
    chairmanTemperature,
    stage2Temperature,
    councilMemberFilters,
    chairmanFilter,
    prompts
  ]);

  // Helper to determine if filters need to switch based on availability
  const isRemoteAvailable = enabledProviders.openrouter || enabledProviders.direct || enabledProviders.groq || enabledProviders.custom;
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

  }, [enabledProviders, chairmanFilter]);

  // Auto-save council configuration with debounce
  const autoSaveTimerRef = useRef(null);
  const isInitialLoadRef = useRef(true);
  const prevCouncilModelsRef = useRef(null);
  const prevChairmanModelRef = useRef(null);

  useEffect(() => {
    // Skip auto-save on initial load
    if (isInitialLoadRef.current) {
      return;
    }

    // Skip if settings haven't loaded yet
    if (!settings) {
      return;
    }

    // Check if council models or chairman actually changed (not just re-rendered)
    const councilModelsStr = JSON.stringify(councilModels);
    const prevCouncilModelsStr = JSON.stringify(prevCouncilModelsRef.current);
    const chairmanChanged = chairmanModel !== prevChairmanModelRef.current;
    const councilChanged = councilModelsStr !== prevCouncilModelsStr;

    if (!councilChanged && !chairmanChanged) {
      return;
    }

    // Update refs
    prevCouncilModelsRef.current = councilModels;
    prevChairmanModelRef.current = chairmanModel;

    // Skip if all values are empty (reset state)
    const hasValidCouncil = councilModels.some(m => m && m.length > 0);
    const hasValidChairman = chairmanModel && chairmanModel.length > 0;
    if (!hasValidCouncil && !hasValidChairman) {
      return;
    }

    // Clear any existing timer
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }

    // Debounce auto-save by 1 second
    autoSaveTimerRef.current = setTimeout(async () => {
      try {
        await api.updateSettings({
          council_models: councilModels,
          chairman_model: chairmanModel
        });
        console.log('Auto-saved council configuration');
      } catch (err) {
        console.error('Failed to auto-save council configuration:', err);
      }
    }, 1000);

    // Cleanup on unmount
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
      }
    };
  }, [councilModels, chairmanModel, settings]);

  // Clear validation errors when chairman is selected
  useEffect(() => {
    if (chairmanModel && validationErrors.chairman) {
      setValidationErrors({});
      setError(null);
    }
  }, [chairmanModel, validationErrors.chairman]);

  const loadSettings = async () => {
    try {
      const data = await api.getSettings();

      // Set settings immediately to show UI
      setSettings(data);

      setSelectedSearchProvider(data.search_provider || 'duckduckgo');
      setSearchKeywordExtraction(data.search_keyword_extraction || 'direct');
      setFullContentResults(data.full_content_results ?? 3);
      setSearchResultCount(data.search_result_count ?? 8);
      setSearchHybridMode(data.search_hybrid_mode ?? true);
      setShowFreeOnly(data.show_free_only ?? false);

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
      const loadedCouncilModels = data.council_models || [];
      const loadedChairmanModel = data.chairman_model || '';
      setCouncilModels(loadedCouncilModels);
      setChairmanModel(loadedChairmanModel);
      setCouncilTemperature(data.council_temperature ?? 0.5);
      setChairmanTemperature(data.chairman_temperature ?? 0.4);
      setStage2Temperature(data.stage2_temperature ?? 0.3);

      // Initialize refs for auto-save tracking (prevents auto-save on initial load)
      prevCouncilModelsRef.current = loadedCouncilModels;
      prevChairmanModelRef.current = loadedChairmanModel;
      // Mark initial load as complete after a short delay to let state settle
      setTimeout(() => {
        isInitialLoadRef.current = false;
      }, 500);

      // Remote/Local filters - load from saved settings
      if (data.council_member_filters) {
        setCouncilMemberFilters(data.council_member_filters);
      }
      if (data.chairman_filter) {
        setChairmanFilter(data.chairman_filter);
      }
      // Ollama Settings
      setOllamaBaseUrl(data.ollama_base_url || 'http://localhost:11434');

      // Custom Endpoint Settings
      if (data.custom_endpoint_name) setCustomEndpointName(data.custom_endpoint_name);
      if (data.custom_endpoint_url) setCustomEndpointUrl(data.custom_endpoint_url);
      // API key is not sent to frontend for security, similar to other keys

      // Prompts
      setPrompts({
        stage1_prompt: data.stage1_prompt || '',
        stage2_prompt: data.stage2_prompt || '',
        stage3_prompt: data.stage3_prompt || '',

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
      if (data.custom_endpoint_url) {
        loadCustomEndpointModels();
      }

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

  const loadCustomEndpointModels = async () => {
    try {
      const data = await api.getCustomEndpointModels();
      if (data.models && data.models.length > 0) {
        const sorted = data.models.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        setCustomEndpointModels(sorted);
      }
    } catch (err) {
      console.warn('Failed to load custom endpoint models:', err);
    }
  };

  const handleTestCustomEndpoint = async () => {
    if (!customEndpointName || !customEndpointUrl) {
      setCustomEndpointTestResult({ success: false, message: 'Please enter a name and URL' });
      return;
    }
    setIsTestingCustomEndpoint(true);
    setCustomEndpointTestResult(null);
    try {
      const result = await api.testCustomEndpoint(customEndpointName, customEndpointUrl, customEndpointApiKey);
      setCustomEndpointTestResult(result);

      // Auto-save if connection succeeds
      if (result.success) {
        await api.updateSettings({
          custom_endpoint_name: customEndpointName,
          custom_endpoint_url: customEndpointUrl,
          custom_endpoint_api_key: customEndpointApiKey || null
        });
        // Reload settings to get the updated state
        const updatedSettings = await api.getSettings();
        setSettings(updatedSettings);
        // Load models from the new endpoint
        loadCustomEndpointModels();
      }
    } catch (err) {
      setCustomEndpointTestResult({ success: false, message: err.message });
    } finally {
      setIsTestingCustomEndpoint(false);
    }
  };

  const handleTestSerper = async () => {
    if (!serperApiKey && !settings.serper_api_key_set) {
      setSerperTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingSerper(true);
    setSerperTestResult(null);
    try {
      // If input is empty but key is configured, pass null to test the saved key
      const keyToTest = serperApiKey || null;
      const result = await api.testSerperKey(keyToTest);
      setSerperTestResult(result);

      // Auto-save API key AND provider selection if validation succeeds
      if (result.success && serperApiKey) {
        // Save both the API key and switch provider to Serper
        await api.updateSettings({ 
          serper_api_key: serperApiKey,
          search_provider: 'serper'
        });
        setSerperApiKey(''); // Clear input after save

        // Reload settings (will now have serper as provider)
        await loadSettings();

        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setSerperTestResult({ success: false, message: 'Test failed' });
    } finally {
      setIsTestingSerper(false);
    }
  };

  const handleTestTavily = async () => {
    if (!tavilyApiKey && !settings.tavily_api_key_set) {
      setTavilyTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingTavily(true);
    setTavilyTestResult(null);
    try {
      // If input is empty but key is configured, pass null to test the saved key
      const keyToTest = tavilyApiKey || null;
      const result = await api.testTavilyKey(keyToTest);
      setTavilyTestResult(result);

      // Auto-save API key AND provider selection if validation succeeds
      if (result.success && tavilyApiKey) {
        // Save both the API key and switch provider to Tavily
        await api.updateSettings({ 
          tavily_api_key: tavilyApiKey,
          search_provider: 'tavily'
        });
        setTavilyApiKey(''); // Clear input after save

        // Reload settings (will now have tavily as provider)
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
    if (!braveApiKey && !settings.brave_api_key_set) {
      setBraveTestResult({ success: false, message: 'Please enter an API key first' });
      return;
    }
    setIsTestingBrave(true);
    setBraveTestResult(null);
    try {
      // If input is empty but key is configured, pass null to test the saved key
      const keyToTest = braveApiKey || null;
      const result = await api.testBraveKey(keyToTest);
      setBraveTestResult(result);

      // Auto-save API key AND provider selection if validation succeeds
      if (result.success && braveApiKey) {
        // Save both the API key and switch provider to Brave
        await api.updateSettings({ 
          brave_api_key: braveApiKey,
          search_provider: 'brave'
        });
        setBraveApiKey(''); // Clear input after save

        // Reload settings (will now have brave as provider)
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
      // Pass empty string if using stored key, backend will handle it
      const result = await api.testProviderKey('groq', groqApiKey || "");
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
    let candidateModels = filteredAvailableModels;

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

    // Apply Updates
    setCouncilModels(newCouncilModels);
    setCouncilMemberFilters(newMemberFilters);

    setChairmanModel(randomChairman.id);
    setChairmanFilter(getFilterForModel(randomChairman.id));

    setSuccess(true);
    setTimeout(() => setSuccess(false), 2000);
  };

  const handleAddCouncilMember = () => {
    const newIndex = councilModels.length;

    // Determine best default filter based on what's available
    let defaultFilter = 'remote';
    const isRemoteAvailable = enabledProviders.openrouter || enabledProviders.direct || enabledProviders.groq || enabledProviders.custom;
    const isLocalAvailable = enabledProviders.ollama && ollamaAvailableModels.length > 0;

    if (!isRemoteAvailable && isLocalAvailable) {
      defaultFilter = 'local';
    }

    // Get models for the chosen filter
    const filtered = filterByRemoteLocal(filteredAvailableModels, defaultFilter);

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
      console.log('Defaults fetched:', defaults);
      if (defaults[key] !== undefined) {
        handlePromptChange(key, defaults[key]);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } else {
        console.warn(`Default for key ${key} not found in defaults`, defaults);
      }
    } catch (err) {
      console.error("Failed to fetch default prompt", err);
      setError("Failed to reset prompt");
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
      // Initialize with 2 empty slots for council
      setCouncilModels(['', '']);
      setChairmanModel('');
      setCouncilTemperature(0.5);
      setChairmanTemperature(0.4);
      setStage2Temperature(0.3);

      // Reset filters to 'remote' default
      // Reset filters to 'remote' default
      setCouncilMemberFilters({ 0: 'remote', 1: 'remote' });
      setChairmanFilter('remote');

      // 3. General Settings Defaults
      setSelectedSearchProvider('duckduckgo');
      setSearchKeywordExtraction('direct');
      setFullContentResults(3);
      setShowFreeOnly(false);
      setOllamaBaseUrl('http://localhost:11434');

      // 4. Reset Prompts to System Defaults (keep these useful)
      const defaults = await api.getDefaultSettings();
      setPrompts({
        stage1_prompt: defaults.stage1_prompt,
        stage2_prompt: defaults.stage2_prompt,
        stage3_prompt: defaults.stage3_prompt,

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
        council_models: ['', ''],
        chairman_model: '',
        council_temperature: 0.5,
        chairman_temperature: 0.4,
        stage2_temperature: 0.3,
        search_query_model: '',
        council_member_filters: { 0: 'remote', 1: 'remote' },
        chairman_filter: 'remote',
        search_query_filter: 'remote',
        stage1_prompt: defaults.stage1_prompt,
        stage2_prompt: defaults.stage2_prompt,
        stage3_prompt: defaults.stage3_prompt,
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
    // Allow if key is entered OR if it's already set (Retest mode)
    if (!apiKey && !settings?.[`${keyField}_set`]) return;

    setValidatingKeys(prev => ({ ...prev, [providerId]: true }));
    setKeyValidationStatus(prev => ({ ...prev, [providerId]: null }));

    try {
      // Pass empty string if using stored key, backend will handle it
      const result = await api.testProviderKey(providerId, apiKey || "");
      setKeyValidationStatus(prev => ({
        ...prev,
        [providerId]: {
          success: result.success,
          message: result.message
        }
      }));

      // Auto-save API key if validation succeeds AND it was a new key
      if (result.success && apiKey) {
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
      search_keyword_extraction: searchKeywordExtraction,
      full_content_results: fullContentResults,
      search_result_count: searchResultCount,
      search_hybrid_mode: searchHybridMode,
      show_free_only: showFreeOnly,

      // Enabled Providers
      enabled_providers: enabledProviders,
      direct_provider_toggles: directProviderToggles,

      // Council Configuration (unified)
      council_models: councilModels,
      chairman_model: chairmanModel,

      // Temperature Settings
      council_temperature: councilTemperature,
      chairman_temperature: chairmanTemperature,
      stage2_temperature: stage2Temperature,

      // Filters
      council_member_filters: councilMemberFilters,
      chairman_filter: chairmanFilter,

      // Ollama Base URL
      ollama_base_url: ollamaBaseUrl,

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
        if (config.search_keyword_extraction) setSearchKeywordExtraction(config.search_keyword_extraction);
        if (config.full_content_results !== undefined) setFullContentResults(config.full_content_results);
        if (config.search_result_count !== undefined) setSearchResultCount(config.search_result_count);
        if (config.search_hybrid_mode !== undefined) setSearchHybridMode(config.search_hybrid_mode);
        if (config.show_free_only !== undefined) setShowFreeOnly(config.show_free_only);

        // Apply Enabled Providers
        if (config.enabled_providers) {
          setEnabledProviders(config.enabled_providers);
        }
        if (config.direct_provider_toggles) {
          setDirectProviderToggles(config.direct_provider_toggles);
        }

        // Apply Council Configuration (unified)
        if (config.council_models) setCouncilModels(config.council_models);
        if (config.chairman_model) setChairmanModel(config.chairman_model);

        // Apply Temperature Settings
        if (config.council_temperature !== undefined) setCouncilTemperature(config.council_temperature);
        if (config.chairman_temperature !== undefined) setChairmanTemperature(config.chairman_temperature);
        if (config.stage2_temperature !== undefined) setStage2Temperature(config.stage2_temperature);

        // Apply Filters
        if (config.council_member_filters) setCouncilMemberFilters(config.council_member_filters);
        if (config.chairman_filter) setChairmanFilter(config.chairman_filter);

        // Apply Ollama Base URL
        if (config.ollama_base_url) setOllamaBaseUrl(config.ollama_base_url);

        // Apply Prompts
        if (config.prompts) {
          setPrompts(prev => ({ ...prev, ...config.prompts }));
        }

        // Validate imported models against all available models
        const allModels = allAvailableModels;
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
    setError(null);
    setSuccess(false);
    setValidationErrors({});

    // Validate council configuration
    const hasCouncilMembers = councilModels.some(m => m && m.length > 0);
    const hasChairman = chairmanModel && chairmanModel.length > 0;

    // If council members are selected but no chairman, show error
    if (hasCouncilMembers && !hasChairman) {
      setValidationErrors({ chairman: true });
      setError('Please select a Chairman to complete the council configuration.');
      
      // Focus on the chairman select and scroll to council section
      setActiveSection('council');
      setTimeout(() => {
        if (chairmanSelectRef.current) {
          chairmanSelectRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
          chairmanSelectRef.current.focus();
        }
      }, 100);
      return;
    }

    setIsSaving(true);

    try {
      const updates = {
        search_provider: selectedSearchProvider,
        search_keyword_extraction: searchKeywordExtraction,
        full_content_results: fullContentResults,
        search_result_count: searchResultCount,
        search_hybrid_mode: searchHybridMode,
        show_free_only: showFreeOnly,

        // Enabled Providers
        enabled_providers: enabledProviders,
        direct_provider_toggles: directProviderToggles,

        // Council Configuration (unified)
        council_models: councilModels,
        chairman_model: chairmanModel,
        council_temperature: councilTemperature,
        chairman_temperature: chairmanTemperature,
        stage2_temperature: stage2Temperature,

        // Remote/Local filters for each selection
        council_member_filters: councilMemberFilters,
        chairman_filter: chairmanFilter,
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
  const allAvailableModels = useMemo(() => {
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

    // Add custom endpoint models if enabled and configured
    if (enabledProviders.custom && customEndpointModels.length > 0) {
      models.push(...customEndpointModels);
    }

    // Deduplicate by model ID (prefer direct connections over OpenRouter for same model)
    // Since direct models are added last, always set to overwrite earlier entries
    const uniqueModels = new Map();
    models.forEach(model => {
      uniqueModels.set(model.id, model);
    });

    return Array.from(uniqueModels.values()).sort((a, b) => (a.name || '').localeCompare(b.name || ''));
  }, [
    enabledProviders,
    availableModels,
    ollamaAvailableModels,
    directAvailableModels,
    customEndpointModels,
    directProviderToggles,
    directKeys,
    settings
  ]);

  // Get filtered models for council member selection (respects free filter)
  const filteredAvailableModels = useMemo(() => {
    const all = allAvailableModels;
    if (!showFreeOnly) return all;

    // Filter logic:
    // 1. If it's an OpenRouter model, checks if it's free.
    // 2. If it's NOT OpenRouter (Direct, Ollama, Custom), keep it visible.
    return all.filter(m => {
      // Check if it's an OpenRouter model
      const isOpenRouter = m.source === 'openrouter' || m.provider === 'OpenRouter' || m.id.startsWith('openrouter:') || m.id.includes('/');

      // If it is OpenRouter, apply the free filter
      if (isOpenRouter) {
        return m.is_free;
      }

      // Otherwise (Direct, Ollama, Custom), always show
      return true;
    });
  }, [allAvailableModels, showFreeOnly]);



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
              <ProviderSettings
                settings={settings}
                // OpenRouter
                openrouterApiKey={openrouterApiKey}
                setOpenrouterApiKey={(val) => { setOpenrouterApiKey(val); setOpenrouterTestResult(null); }}
                handleTestOpenRouter={handleTestOpenRouter}
                isTestingOpenRouter={isTestingOpenRouter}
                openrouterTestResult={openrouterTestResult}
                // Groq
                groqApiKey={groqApiKey}
                setGroqApiKey={(val) => { setGroqApiKey(val); setGroqTestResult(null); }}
                handleTestGroq={handleTestGroq}
                isTestingGroq={isTestingGroq}
                groqTestResult={groqTestResult}
                // Ollama
                ollamaBaseUrl={ollamaBaseUrl}
                setOllamaBaseUrl={(val) => { setOllamaBaseUrl(val); setOllamaTestResult(null); }}
                handleTestOllama={handleTestOllama}
                isTestingOllama={isTestingOllama}
                ollamaTestResult={ollamaTestResult}
                ollamaStatus={ollamaStatus}
                loadOllamaModels={loadOllamaModels}
                // Direct
                directKeys={directKeys}
                setDirectKeys={setDirectKeys}
                handleTestDirectKey={handleTestDirectKey}
                validatingKeys={validatingKeys}
                keyValidationStatus={keyValidationStatus}
                // Custom Endpoint
                customEndpointName={customEndpointName}
                setCustomEndpointName={(val) => { setCustomEndpointName(val); setCustomEndpointTestResult(null); }}
                customEndpointUrl={customEndpointUrl}
                setCustomEndpointUrl={(val) => { setCustomEndpointUrl(val); setCustomEndpointTestResult(null); }}
                customEndpointApiKey={customEndpointApiKey}
                setCustomEndpointApiKey={(val) => { setCustomEndpointApiKey(val); setCustomEndpointTestResult(null); }}
                handleTestCustomEndpoint={handleTestCustomEndpoint}
                isTestingCustomEndpoint={isTestingCustomEndpoint}
                customEndpointTestResult={customEndpointTestResult}
                customEndpointModels={customEndpointModels}
              />
            )}

            {/* COUNCIL CONFIGURATION */}
            {activeSection === 'council' && (
              <CouncilConfig
                settings={settings}
                // State
                enabledProviders={enabledProviders}
                setEnabledProviders={setEnabledProviders}
                directProviderToggles={directProviderToggles}
                setDirectProviderToggles={setDirectProviderToggles}
                showFreeOnly={showFreeOnly}
                setShowFreeOnly={setShowFreeOnly}
                isLoadingModels={isLoadingModels}
                rateLimitWarning={rateLimitWarning}
                councilModels={councilModels}
                councilMemberFilters={councilMemberFilters}
                chairmanModel={chairmanModel}
                setChairmanModel={setChairmanModel}
                chairmanFilter={chairmanFilter}
                setChairmanFilter={setChairmanFilter}
                councilTemperature={councilTemperature}
                setCouncilTemperature={setCouncilTemperature}
                chairmanTemperature={chairmanTemperature}
                setChairmanTemperature={setChairmanTemperature}
                // Data
                allModels={allAvailableModels}
                filteredModels={filteredAvailableModels}
                ollamaAvailableModels={ollamaAvailableModels}
                customEndpointName={customEndpointName}
                customEndpointUrl={customEndpointUrl}
                // Callbacks
                handleFeelingLucky={handleFeelingLucky}
                handleMemberFilterChange={handleMemberFilterChange}
                handleCouncilModelChange={handleCouncilModelChange}
                handleRemoveCouncilMember={handleRemoveCouncilMember}
                handleAddCouncilMember={handleAddCouncilMember}
                setActiveSection={setActiveSection}
                setActivePromptTab={setActivePromptTab}
                // Validation
                validationErrors={validationErrors}
                chairmanSelectRef={chairmanSelectRef}
              />
            )}

            {/* SYSTEM PROMPTS */}
            {activeSection === 'prompts' && (
              <PromptSettings
                prompts={prompts}
                handlePromptChange={handlePromptChange}
                handleResetPrompt={handleResetPrompt}
                activePromptTab={activePromptTab}
                setActivePromptTab={setActivePromptTab}
                stage2Temperature={stage2Temperature}
                setStage2Temperature={setStage2Temperature}
              />
            )}

            {/* SEARCH PROVIDERS (New Section) */}
            {activeSection === 'search' && (
              <SearchSettings
                settings={settings}
                selectedSearchProvider={selectedSearchProvider}
                setSelectedSearchProvider={setSelectedSearchProvider}
                // Serper (Google)
                serperApiKey={serperApiKey}
                setSerperApiKey={setSerperApiKey}
                handleTestSerper={handleTestSerper}
                isTestingSerper={isTestingSerper}
                serperTestResult={serperTestResult}
                setSerperTestResult={setSerperTestResult}
                // Tavily
                tavilyApiKey={tavilyApiKey}
                setTavilyApiKey={setTavilyApiKey}
                handleTestTavily={handleTestTavily}
                isTestingTavily={isTestingTavily}
                tavilyTestResult={tavilyTestResult}
                setTavilyTestResult={setTavilyTestResult}
                // Brave
                braveApiKey={braveApiKey}
                setBraveApiKey={setBraveApiKey}
                handleTestBrave={handleTestBrave}
                isTestingBrave={isTestingBrave}
                braveTestResult={braveTestResult}
                setBraveTestResult={setBraveTestResult}
                // Other Settings
                fullContentResults={fullContentResults}
                setFullContentResults={setFullContentResults}
                searchKeywordExtraction={searchKeywordExtraction}
                setSearchKeywordExtraction={setSearchKeywordExtraction}
                searchResultCount={searchResultCount}
                setSearchResultCount={setSearchResultCount}
                searchHybridMode={searchHybridMode}
                setSearchHybridMode={setSearchHybridMode}
              />
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

      {
        showResetConfirm && (
          <div className="settings-overlay confirmation-overlay" onClick={() => setShowResetConfirm(false)}>
            <div className="settings-modal confirmation-modal" onClick={e => e.stopPropagation()}>
              <div className="settings-header">
                <h2>Confirm Reset</h2>
              </div>
              <div className="settings-content confirmation-content" style={{ padding: '20px 24px' }}>
                <p style={{ marginBottom: '16px' }}>Are you sure you want to reset to defaults?</p>
                <div className="confirmation-details" style={{ padding: '16px 20px' }}>
                  <p><strong>This will reset:</strong></p>
                  <ul style={{ margin: '12px 0', lineHeight: '1.8' }}>
                    <li>Provider toggles → All disabled</li>
                    <li>Model selections → Cleared</li>
                    <li>Council size → Reset to 2 members</li>
                    <li>Temperatures → Defaults (0.5 / 0.4 / 0.3)</li>
                    <li>System prompts → Defaults</li>
                    <li>Search provider → DuckDuckGo</li>
                    <li>Jina fetch count → 3</li>
                    <li>Ollama URL → localhost:11434</li>
                  </ul>
                  <p className="confirmation-safe" style={{ marginTop: '14px' }}>✓ API keys will be PRESERVED</p>
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
        )
      }
    </div >
  );
}