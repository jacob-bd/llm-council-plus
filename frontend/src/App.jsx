import { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import Settings from './components/Settings';
import { api } from './api';
import './App.css';

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
  const [executionMode, setExecutionMode] = useState('full');
  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);
  const isInitialMount = useRef(true);

  // Check initial configuration on mount
  useEffect(() => {
    checkInitialSetup();
  }, []);

  const checkInitialSetup = async () => {
    try {
      // 1. Get Settings to check for API keys
      const settings = await api.getSettings();

      // Load execution mode preference
      setExecutionMode(settings.execution_mode || 'full');
      setSearchProvider(settings.search_provider || 'duckduckgo');

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
      const models = settings.council_models || [];
      const chairman = settings.chairman_model || '';

      setCouncilModels(models);
      setChairmanModel(chairman);

      const hasCouncilMembers = models.some(m => m && m.trim() !== '');
      const hasChairman = chairman && chairman.trim() !== '';
      setCouncilConfigured(hasCouncilMembers && hasChairman);

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
      const models = settings.council_models || [];
      const chairman = settings.chairman_model || '';

      setCouncilModels(models);
      setChairmanModel(chairman);
      setSearchProvider(settings.search_provider || 'duckduckgo');

      const hasCouncilMembers = models.some(m => m && m.trim() !== '');
      const hasChairman = chairman && chairman.trim() !== '';
      setCouncilConfigured(hasCouncilMembers && hasChairman);
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
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    // Check if there's already an empty/unused conversation
    const existingEmpty = conversations.find(conv => !conv.title && conv.message_count === 0);

    if (existingEmpty) {
      // Reuse the existing empty conversation instead of creating a new one
      setCurrentConversationId(existingEmpty.id);
      return;
    }

    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
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

  const handleSendMessage = async (content, webSearch) => {
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
        { content, webSearch, executionMode },
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

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        onOpenSettings={() => setShowSettings(true)}
        isLoading={isLoading}
        onAbort={handleAbort}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        onAbort={handleAbort}
        isLoading={isLoading}
        councilConfigured={councilConfigured}
        councilModels={councilModels}
        chairmanModel={chairmanModel}
        searchProvider={searchProvider}
        onOpenSettings={handleOpenSettings}
        executionMode={executionMode}
        onExecutionModeChange={setExecutionMode}
      />
      {showSettings && (
        <Settings
          onClose={handleSettingsClose}
          ollamaStatus={ollamaStatus}
          onRefreshOllama={testOllamaConnection}
          initialSection={settingsInitialSection}
        />
      )}
    </div>
  );
}

export default App;
