import { Suspense, lazy, useState, useEffect, useRef, useCallback, Component } from 'react';
import Sidebar from './components/Sidebar';
import { api, DEFAULT_EXECUTION_MODE, buildAvailableSearchProviders } from './api';
import './App.css';
import './components/StageCopyButtons.css';
import './ModeToggle.css';

const ChatInterface = lazy(() => import('./components/ChatInterface'));
const Settings = lazy(() => import('./components/Settings'));
const LandingPage = lazy(() => import('./components/LandingPage'));

function AppLoadingFallback() {
  return (
    <div className="app-loading" role="status" aria-live="polite">
      Loading...
    </div>
  );
}

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-loading" role="alert">
          Failed to load. Please refresh the page.
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsInitialSection, setSettingsInitialSection] = useState('llm_keys');
  const [ollamaStatus, setOllamaStatus] = useState({
    connected: false,
    lastConnected: null,
    testing: false
  });
  const [councilConfigured, setCouncilConfigured] = useState(true); // Assume configured until checked
  const [councilModels, setCouncilModels] = useState([]);
  const [chairmanModel, setChairmanModel] = useState(null);
  const [searchProvider, setSearchProvider] = useState('duckduckgo');
  const [availableSearchProviders, setAvailableSearchProviders] = useState([{ id: 'duckduckgo', name: 'DuckDuckGo' }]);
  const [executionMode, setExecutionMode] = useState(DEFAULT_EXECUTION_MODE);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [appMode, setAppMode] = useState(null); // null shows landing page
  const abortControllerRef = useRef(null);
  const advisorAbortControllerRef = useRef(null);
  const requestIdRef = useRef(0);
  const isInitialMount = useRef(true);
  const conversationVersionRef = useRef(0);
  const skipLoadForIdRef = useRef(null);

  const computeCouncilConfigured = useCallback((models, chairman, mode) => {
    const members = (models || []).filter((m) => m && m.trim());
    if (members.length < 1) return false;
    if (mode === 'full') {
      return !!(chairman && chairman.trim());
    }
    return true;
  }, []);

  const handleCouncilChange = useCallback(async ({ councilModels: nextModels, chairmanModel: nextChairman }) => {
    const filtered = (nextModels || []).filter((m) => m && m.trim());
    const chairman = nextChairman || '';
    setCouncilModels(filtered);
    setChairmanModel(chairman);
    setCouncilConfigured(computeCouncilConfigured(filtered, chairman, executionMode));
    try {
      await api.updateSettings({
        council_models: filtered,
        chairman_model: chairman,
      });
    } catch (err) {
      console.error('Failed to save council lineup:', err);
    }
  }, [computeCouncilConfigured, executionMode]);

  useEffect(() => {
    setCouncilConfigured(computeCouncilConfigured(councilModels, chairmanModel, executionMode));
  }, [councilModels, chairmanModel, executionMode, computeCouncilConfigured]);

  // Check initial configuration on mount
  useEffect(() => {
    checkInitialSetup();
  }, []);

  const checkInitialSetup = async () => {
    try {
      // 1. Get Settings to check for API keys
      const settings = await api.getSettings();

      // Load execution mode preference
      setExecutionMode(settings.execution_mode || DEFAULT_EXECUTION_MODE);
      setSearchProvider(settings.search_provider || 'duckduckgo');

      setAvailableSearchProviders(buildAvailableSearchProviders(settings));

      const hasApiKey = settings.openrouter_api_key_set ||
        settings.groq_api_key_set ||
        settings.openai_api_key_set ||
        settings.anthropic_api_key_set ||
        settings.google_api_key_set ||
        settings.mistral_api_key_set ||
        settings.deepseek_api_key_set;

      // 2. Test Ollama Connection
      // We do this regardless to update the status indicator
      const ollamaUrl = settings.ollama_base_url || 'http://localhost:11434';
      setOllamaStatus(prev => ({ ...prev, testing: true }));

      let isOllamaConnected = false;
      try {
        const result = await api.testOllamaConnection(ollamaUrl);
        isOllamaConnected = result.success;

        if (result.success) {
          setOllamaStatus({
            connected: true,
            lastConnected: new Date().toLocaleString(),
            testing: false
          });
        } else {
          setOllamaStatus({ connected: false, lastConnected: null, testing: false });
        }
      } catch (err) {
        console.error('Ollama initial test failed:', err);
        setOllamaStatus({ connected: false, lastConnected: null, testing: false });
      }

      // 3. Check if council is configured (has models selected)
      const models = (settings.council_models || []).filter((m) => m && m.trim());
      const chairman = settings.chairman_model || '';

      setCouncilModels(models);
      setChairmanModel(chairman);

      setCouncilConfigured(computeCouncilConfigured(models, chairman, settings.execution_mode || DEFAULT_EXECUTION_MODE));

      // 4. If no providers are configured, open settings
      if (!hasApiKey && !isOllamaConnected) {
        setShowSettings(true);
      }

    } catch (error) {
      console.error('Failed to check initial setup:', error);
    }
  };

  // Re-check council configuration when settings close
  const handleSettingsClose = async () => {
    setShowSettings(false);
    try {
      const settings = await api.getSettings();
      const models = (settings.council_models || []).filter((m) => m && m.trim());
      const chairman = settings.chairman_model || '';

      setCouncilModels(models);
      setChairmanModel(chairman);
      setSearchProvider(settings.search_provider || 'duckduckgo');
      setAvailableSearchProviders(buildAvailableSearchProviders(settings));
      setExecutionMode(settings.execution_mode || DEFAULT_EXECUTION_MODE);

      setCouncilConfigured(computeCouncilConfigured(
        models,
        chairman,
        settings.execution_mode || DEFAULT_EXECUTION_MODE
      ));
    } catch (error) {
      console.error('Error after closing settings:', error);
    }
  };

  const handleOpenSettings = (section = 'council') => {
    setSettingsInitialSection(section || 'council');
    setShowSettings(true);
  };

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Auto-save execution mode preference when changed
  useEffect(() => {
    // Skip saving on initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    const saveExecutionMode = async () => {
      try {
        await api.updateSettings({ execution_mode: executionMode });
      } catch (error) {
        console.error('Failed to save execution mode:', error);
      }
    };

    saveExecutionMode();
  }, [executionMode]);

  const testOllamaConnection = async (customUrl = null) => {
    try {
      setOllamaStatus(prev => ({ ...prev, testing: true }));

      // Use custom URL if provided, otherwise get from settings
      let urlToTest = customUrl;
      if (!urlToTest) {
        const settings = await api.getSettings();
        urlToTest = settings.ollama_base_url;
      }

      if (!urlToTest) {
        setOllamaStatus({ connected: false, lastConnected: null, testing: false });
        return;
      }

      const result = await api.testOllamaConnection(urlToTest);

      if (result.success) {
        setOllamaStatus({
          connected: true,
          lastConnected: new Date().toLocaleString(),
          testing: false
        });
      } else {
        setOllamaStatus({ connected: false, lastConnected: null, testing: false });
      }
    } catch (error) {
      console.error('Ollama connection test failed:', error);
      setOllamaStatus({ connected: false, lastConnected: null, testing: false });
    }
  };

  // Load conversation details when selected
  useEffect(() => {
    if (currentConversationId) {
      if (skipLoadForIdRef.current === currentConversationId) {
        skipLoadForIdRef.current = null;
        return;
      }
      // Capture the current version — if an optimistic update bumps it
      // before the fetch resolves, the stale response is discarded.
      // This is StrictMode-safe: double-invocation just fires two loads,
      // both of which will be stale if a debate was started.
      const versionAtStart = conversationVersionRef.current;
      loadConversation(currentConversationId, versionAtStart);
    }
  }, [currentConversationId]);

  const loadConversations = async (retryCount = 0) => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
      // Retry up to 3 times with increasing delays (1s, 2s, 3s)
      if (retryCount < 3) {
        setTimeout(() => loadConversations(retryCount + 1), (retryCount + 1) * 1000);
      }
    }
  };

  const loadConversation = async (id, expectedVersion) => {
    try {
      const conv = await api.getConversation(id);
      // Only apply if no newer optimistic update has occurred since we started
      if (conversationVersionRef.current === expectedVersion) {
        setCurrentConversation(conv);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const abortAllStreams = () => {
    if (advisorAbortControllerRef.current) {
      advisorAbortControllerRef.current.abort();
      advisorAbortControllerRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleNewConversation = async () => {
    abortAllStreams();
    setIsLoading(false);
    setAppMode('council');

    // Reuse an empty council conversation only (not advisor drafts)
    const existingEmpty = conversations.find(
      (conv) =>
        conv.mode !== 'advisors'
        && (!conv.title || conv.title === 'New Conversation')
        && conv.message_count === 0
    );

    if (existingEmpty) {
      setCurrentConversation(null);
      setCurrentConversationId(existingEmpty.id);
      return;
    }

    try {
      const newConv = await api.createConversation({ mode: 'council' });
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0, mode: 'council' },
        ...conversations,
      ]);
      setCurrentConversation(null);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
    // Auto-switch mode based on conversation mode
    const conv = conversations.find(c => c.id === id);
    if (conv?.mode === 'advisors') {
      setAppMode('advisors');
    } else {
      setAppMode('council');
    }
  };

  const handleDeleteConversation = async (id) => {
    try {
      await api.deleteConversation(id);
      // Remove from local state
      setConversations(conversations.filter(c => c.id !== id));
      // If we deleted the current conversation, clear it
      if (id === currentConversationId) {
        setCurrentConversationId(null);
        setCurrentConversation(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const handleAbort = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      // Don't set to null here - let the request handler clean up
      // This prevents race conditions with rapid clicks
      setIsLoading(false);
    }
  };

  const handleStartDebate = async (options) => {
    try {
      // Clean up any existing empty conversation before creating a new one
      const existingEmpty = conversations.find(conv => (!conv.title || conv.title === 'New Conversation') && conv.message_count === 0);
      if (existingEmpty) {
        try {
          await api.deleteConversation(existingEmpty.id);
          setConversations(prev => prev.filter(c => c.id !== existingEmpty.id));
        } catch (e) {
          console.error("Failed to cleanup empty conversation", e);
        }
      }

      const newConv = await api.createConversation({ mode: 'advisors' });
      const convId = newConv.id;

      setConversations((prev) => [
        { id: convId, created_at: newConv.created_at, message_count: 0, mode: 'advisors', title: 'New Conversation' },
        ...prev,
      ]);
      // Bump version so any in-flight or upcoming loadConversation calls
      // from the useEffect won't overwrite our optimistic state.
      conversationVersionRef.current++;
      skipLoadForIdRef.current = convId;
      setCurrentConversationId(convId);
      setAppMode('advisors');

      const userMessage = { role: 'user', content: options.question };
      const debateMessage = {
        role: 'assistant',
        type: 'advisor_debate',
        mode: 'advisors',
        isRunning: true,
        currentRound: 0,
        maxRounds: options.maxRounds || 3,
        question: options.question,
        webSearch: options.searchProvider || null,
        personas: [],
        rounds: [],
        verdict: null,
        tiebreaker: null,
        consensusReached: false,
        error: null,
      };

      setCurrentConversation({
        id: convId,
        messages: [userMessage, debateMessage],
      });

      advisorAbortControllerRef.current = new AbortController();
      setIsLoading(true);

      await api.sendDebateStream(
        convId,
        options,
        (eventType, event) => {
          switch (eventType) {
            case 'advisor_debate_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  personas: event.data?.personas || [],
                  maxRounds: event.data?.max_rounds || lastMsg.maxRounds,
                };
                return { ...prev, messages };
              });
              break;

            case 'advisor_round_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  currentRound: event.round || event.data?.round_number || lastMsg.currentRound,
                };
                return { ...prev, messages };
              });
              break;

            case 'advisor_response':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                const rounds = [...(lastMsg.rounds || [])];
                const roundIndex = (event.round || 1) - 1;
                if (!rounds[roundIndex]) {
                  rounds[roundIndex] = { round: event.round, responses: [], complete: false };
                }
                rounds[roundIndex] = {
                  ...rounds[roundIndex],
                  responses: [...rounds[roundIndex].responses, event.data],
                };
                messages[messages.length - 1] = { ...lastMsg, rounds };
                return { ...prev, messages };
              });
              break;

            case 'advisor_round_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                const rounds = [...(lastMsg.rounds || [])];
                const roundIndex = (event.round || 1) - 1;
                if (!rounds[roundIndex]) {
                  rounds[roundIndex] = { round: event.round, responses: [], complete: false };
                }
                rounds[roundIndex] = {
                  ...rounds[roundIndex],
                  complete: true,
                  consensusReached: event.consensus_reached || false,
                };
                messages[messages.length - 1] = {
                  ...lastMsg,
                  rounds,
                  consensusReached: event.consensus_reached || false,
                };
                return { ...prev, messages };
              });
              break;

            case 'advisor_verdict':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  verdict: event.data || event,
                };
                return { ...prev, messages };
              });
              break;

            case 'advisor_tiebreaker':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  tiebreaker: event.data || event,
                };
                return { ...prev, messages };
              });
              break;

            case 'advisor_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = { ...lastMsg, isRunning: false };
                return { ...prev, messages };
              });
              setIsLoading(false);
              break;

            case 'advisor_error':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  isRunning: false,
                  error: event.message || 'Advisor debate failed',
                };
                return { ...prev, messages };
              });
              setIsLoading(false);
              break;

            default:
              break;
          }
        },
        advisorAbortControllerRef.current.signal
      );
    } catch (error) {
      if (error.name === 'AbortError') {
        setCurrentConversation((prev) => {
          if (!prev || prev.messages.length < 2) return prev;
          const messages = [...prev.messages];
          const lastMsg = messages[messages.length - 1];
          if (lastMsg.type === 'advisor_debate') {
            messages[messages.length - 1] = { ...lastMsg, isRunning: false, aborted: true };
          }
          return { ...prev, messages };
        });
        setIsLoading(false);
        return;
      }
      console.error('Failed to start debate:', error);
      // Surface the error to the user instead of showing a blank screen
      setCurrentConversation((prev) => {
        if (!prev?.messages?.length) return prev;
        const messages = [...prev.messages];
        const lastMsg = messages[messages.length - 1];
        if (lastMsg.type === 'advisor_debate') {
          messages[messages.length - 1] = {
            ...lastMsg,
            isRunning: false,
            error: error.message || 'Failed to start debate. Please try again.',
          };
        }
        return { ...prev, messages };
      });
      setIsLoading(false);
    } finally {
      advisorAbortControllerRef.current = null;
      loadConversations();
    }
  };

  const handleSendMessage = async (content, searchProvider) => {
    if (!currentConversationId) return;

    // Assign unique ID to this request to prevent race conditions
    const currentRequestId = ++requestIdRef.current;

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();

    setIsLoading(true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          search: false,
          stage1: false,
          stage2: false,
          stage3: false,
        },
        timers: {
          stage1Start: null,
          stage1End: null,
          stage2Start: null,
          stage2End: null,
          stage3Start: null,
          stage3End: null,
        },
        progress: {
          stage1: { count: 0, total: 0, currentModel: null },
          stage2: { count: 0, total: 0, currentModel: null }
        }
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(
        currentConversationId,
        {
          content,
          searchProvider,
          executionMode,
          councilModels,
          chairmanModel: executionMode === 'full' ? chairmanModel : undefined,
        },
        (eventType, event) => {
          switch (eventType) {
            case 'search_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  loading: {
                    ...lastMsg.loading,
                    search: true
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'search_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  loading: {
                    ...lastMsg.loading,
                    search: false
                  },
                  metadata: {
                    ...lastMsg.metadata,
                    search_query: event.data.search_query,
                    extracted_query: event.data.extracted_query,
                    search_context: event.data.search_context,
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage1_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  loading: {
                    ...lastMsg.loading,
                    stage1: true
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage1Start: Date.now()
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage1_init':
              console.log('DEBUG: Received stage1_init', event);
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  progress: {
                    ...lastMsg.progress,
                    stage1: {
                      count: 0,
                      total: event.total,
                      currentModel: null
                    }
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage1_progress':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                // Immutable update for stage1
                const updatedStage1 = lastMsg.stage1 ? [...lastMsg.stage1, event.data] : [event.data];
                const updatedLastMsg = {
                  ...lastMsg,
                  progress: {
                    ...lastMsg.progress,
                    stage1: {
                      count: event.count,
                      total: event.total,
                      currentModel: event.data.model
                    }
                  },
                  stage1: updatedStage1
                };

                messages[messages.length - 1] = updatedLastMsg;

                return { ...prev, messages };
              });
              break;

            case 'stage1_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                // Immutable update to prevent React rendering issues
                const updatedLastMsg = {
                  ...lastMsg,
                  stage1: event.data,
                  loading: {
                    ...lastMsg.loading,
                    stage1: false
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage1End: Date.now()
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage2_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  loading: {
                    ...lastMsg.loading,
                    stage2: true
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage2Start: Date.now()
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage2_init':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  progress: {
                    ...lastMsg.progress,
                    stage2: {
                      count: 0,
                      total: event.total,
                      currentModel: null
                    }
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage2_progress':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                // Immutable update for stage2
                const updatedStage2 = lastMsg.stage2 ? [...lastMsg.stage2, event.data] : [event.data];
                const updatedLastMsg = {
                  ...lastMsg,
                  progress: {
                    ...lastMsg.progress,
                    stage2: {
                      count: event.count,
                      total: event.total,
                      currentModel: event.data.model
                    }
                  },
                  stage2: updatedStage2
                };

                messages[messages.length - 1] = updatedLastMsg;

                return { ...prev, messages };
              });
              break;

            case 'stage2_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                // Immutable update to prevent React rendering issues
                const updatedLastMsg = {
                  ...lastMsg,
                  stage2: event.data,
                  loading: {
                    ...lastMsg.loading,
                    stage2: false
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage2End: Date.now()
                  },
                  metadata: {
                    ...lastMsg.metadata,
                    ...event.metadata
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage3_start':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                const updatedLastMsg = {
                  ...lastMsg,
                  loading: {
                    ...lastMsg.loading,
                    stage3: true
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage3Start: Date.now()
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              break;

            case 'stage3_complete':
              setCurrentConversation((prev) => {
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];

                // Immutable update to prevent React rendering issues
                const updatedLastMsg = {
                  ...lastMsg,
                  stage3: event.data,
                  loading: {
                    ...lastMsg.loading,
                    stage3: false
                  },
                  timers: {
                    ...lastMsg.timers,
                    stage3End: Date.now()
                  }
                };

                messages[messages.length - 1] = updatedLastMsg;
                return { ...prev, messages };
              });
              // Hide loading indicator once final answer is shown
              setIsLoading(false);
              break;

            case 'title_complete':
              // Reload conversations to get updated title
              loadConversations();
              break;

            case 'complete':
              // Stream complete, reload conversations list
              loadConversations();
              setIsLoading(false);
              break;

            case 'error':
              console.error('Stream error:', event.message);
              setCurrentConversation((prev) => {
                if (!prev || prev.messages.length === 0) return prev;
                const messages = [...prev.messages];
                const lastMsg = messages[messages.length - 1];
                if (lastMsg.role === 'assistant') {
                  messages[messages.length - 1] = {
                    ...lastMsg,
                    error: event.message || 'The council request failed.',
                    loading: {
                      search: false,
                      stage1: false,
                      stage2: false,
                      stage3: false,
                    },
                  };
                }
                return { ...prev, messages };
              });
              setIsLoading(false);
              break;

            default:
              console.log('Unknown event type:', eventType);
          }
        }, abortControllerRef.current?.signal);
    } catch (error) {
      // Handle aborted requests - mark message as aborted
      if (error.name === 'AbortError') {
        console.log('Request aborted');
        // Mark the assistant message as aborted and stop timers
        setCurrentConversation((prev) => {
          if (!prev || prev.messages.length < 2) return prev;
          const messages = [...prev.messages];
          const lastMsg = messages[messages.length - 1];
          if (lastMsg.role === 'assistant') {
            const now = Date.now();
            messages[messages.length - 1] = {
              ...lastMsg,
              aborted: true,
              loading: {
                search: false,
                stage1: false,
                stage2: false,
                stage3: false,
              },
              timers: {
                ...lastMsg.timers,
                // Stop any running timers
                stage1End: lastMsg.timers?.stage1Start && !lastMsg.timers?.stage1End ? now : lastMsg.timers?.stage1End,
                stage2End: lastMsg.timers?.stage2Start && !lastMsg.timers?.stage2End ? now : lastMsg.timers?.stage2End,
                stage3End: lastMsg.timers?.stage3Start && !lastMsg.timers?.stage3End ? now : lastMsg.timers?.stage3End,
              }
            };
          }
          return { ...prev, messages };
        });
        setIsLoading(false);
        return;
      }
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    } finally {
      // Only clear the controller if this is still the current request
      // This prevents race conditions if user rapidly sends multiple messages
      if (requestIdRef.current === currentRequestId) {
        abortControllerRef.current = null;
      }
      // Reload conversations to ensure title/messages are synced, even if aborted
      loadConversations();
    }
  };

  // Mobile sidebar handlers
  const handleMobileSelectConversation = (id) => {
    handleSelectConversation(id);
    setSidebarOpen(false); // Close sidebar on mobile after selection
  };

  const handleMobileNewConversation = async () => {
    await handleNewConversation();
    setSidebarOpen(false); // Close sidebar on mobile after creating new conversation
  };

  const resetAppState = (mode) => {
    abortAllStreams();
    setIsLoading(false);
    setCurrentConversationId(null);
    setCurrentConversation(null);
    setAppMode(mode);
    setSidebarOpen(false);
  };

  const handleMobileNewAdvisors = () => resetAppState('advisors');

  const handleMobileOpenSettings = () => {
    setShowSettings(true);
    setSidebarOpen(false); // Close sidebar on mobile
  };

  return (
    <div className="app">
      {/* Mobile hamburger menu button */}
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open menu"
      >
        <span className="hamburger-icon"></span>
      </button>

      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleMobileSelectConversation}
        onNewConversation={handleMobileNewConversation}
        onNewAdvisors={handleMobileNewAdvisors}
        onDeleteConversation={handleDeleteConversation}
        onOpenSettings={handleMobileOpenSettings}
        isLoading={isLoading}
        onAbort={handleAbort}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onGoHome={() => resetAppState(null)}
      />

      <div className="main-area">
        <AppErrorBoundary>
          <Suspense fallback={<AppLoadingFallback />}>
            {appMode === null && !currentConversationId ? (
              <LandingPage onSelectMode={(m) => setAppMode(m)} />
            ) : (
              <ChatInterface
                conversation={currentConversation}
                onSendMessage={handleSendMessage}
                onAbort={handleAbort}
                isLoading={isLoading}
                councilConfigured={councilConfigured}
                councilModels={councilModels}
                chairmanModel={chairmanModel}
                searchProvider={searchProvider}
                availableSearchProviders={availableSearchProviders}
                onOpenSettings={handleOpenSettings}
                executionMode={executionMode}
                onExecutionModeChange={setExecutionMode}
                mode={appMode}
                onStartDebate={handleStartDebate}
                onNewConversation={handleNewConversation}
                onCouncilChange={handleCouncilChange}
              />
            )}
          </Suspense>
        </AppErrorBoundary>
      </div>

      {showSettings && (
        <AppErrorBoundary>
          <Suspense fallback={<AppLoadingFallback />}>
            <Settings
              onClose={handleSettingsClose}
              ollamaStatus={ollamaStatus}
              onRefreshOllama={testOllamaConnection}
              initialSection={settingsInitialSection}
            />
          </Suspense>
        </AppErrorBoundary>
      )}
    </div>
  );
}

export default App;
