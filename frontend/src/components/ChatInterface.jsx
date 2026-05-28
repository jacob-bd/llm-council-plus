import StageTimer from './StageTimer';
import { useState, useEffect, useLayoutEffect, useRef } from 'react';
import SearchContext from './SearchContext';
import Stage1, { Stage1Skeleton } from './Stage1';
import Stage2, { Stage2Skeleton } from './Stage2';
import Stage3, { Stage3Skeleton } from './Stage3';
import CouncilGrid from './CouncilGrid';
import CouncilSetup from './CouncilSetup';
import ExecutionModeToggle from './ExecutionModeToggle';
import DebateView from './DebateView';
import AdvisorSetup from './AdvisorSetup';
import MarkdownContent from './MarkdownContent';
import './ChatInterface.css';

function hasStage1Results(msg) {
    return Array.isArray(msg.stage1) && msg.stage1.length > 0;
}

function hasStage2Results(msg) {
    return Array.isArray(msg.stage2) && msg.stage2.length > 0;
}

function hasStage2Started(msg) {
    return Boolean(msg.loading?.stage2 || hasStage2Results(msg));
}

function shouldShowStage1CouncilGrid(msg) {
    return msg.loading?.stage1 || (hasStage1Results(msg) && !hasStage2Started(msg));
}

function shouldShowStage1Results(msg) {
    return msg.loading?.stage1 || hasStage1Results(msg);
}

function getDeliberationScrollPhase(msg) {
    if (!msg || msg.role !== 'assistant') return 'idle';
    if (msg.loading?.stage3 || msg.stage3) return 'stage3';
    if (hasStage2Started(msg)) return 'stage2';
    if (msg.loading?.stage1 || hasStage1Results(msg)) return 'stage1';
    if (msg.loading?.search) return 'search';
    return 'idle';
}

function renderStage1Content(msg) {
    if (!shouldShowStage1Results(msg)) return null;
    if (msg.loading?.stage1 && !hasStage1Results(msg)) return <Stage1Skeleton />;
    if (!hasStage1Results(msg)) return null;
    return (
        <Stage1
            responses={msg.stage1}
            startTime={msg.timers?.stage1Start}
            endTime={msg.timers?.stage1End}
        />
    );
}

function isCouncilTurnPending(msg, isActiveTurn, isLoading) {
    if (!isActiveTurn || !isLoading || msg.error || msg.aborted) return false;
    if (msg.loading?.search || msg.loading?.stage1 || msg.loading?.stage2 || msg.loading?.stage3) {
        return false;
    }
    if (hasStage1Results(msg) || hasStage2Results(msg) || msg.stage3) return false;
    if (msg.metadata?.search_context) return false;
    return true;
}

export default function ChatInterface({
    conversation,
    onSendMessage,
    onAbort,
    isLoading,
    councilConfigured,
    onOpenSettings,
    councilModels = [],
    chairmanModel = null,
    executionMode,
    onExecutionModeChange,
    searchProvider = 'duckduckgo',
    availableSearchProviders = [{ id: 'duckduckgo', name: 'DuckDuckGo' }],
    mode = 'council',
    onStartDebate,
    onNewConversation,
    onCouncilChange,
}) {
    const [input, setInput] = useState('');
    const [activeSearchProvider, setActiveSearchProvider] = useState(null);
    const [searchPopoverOpen, setSearchPopoverOpen] = useState(false);
    const searchPopoverRef = useRef(null);
    const messagesEndRef = useRef(null);
    const messagesContainerRef = useRef(null);
    const stage2AnchorRef = useRef(null);
    const stage3AnchorRef = useRef(null);
    const prevScrollPhaseRef = useRef(null);

    useLayoutEffect(() => {
        if (!messagesContainerRef.current || !conversation?.messages?.length) return;

        const container = messagesContainerRef.current;
        const lastMsg = conversation.messages[conversation.messages.length - 1];
        const phase = getDeliberationScrollPhase(lastMsg);
        const prevPhase = prevScrollPhaseRef.current;
        prevScrollPhaseRef.current = phase;

        const scrollAnchors = {
            'stage1->stage2': stage2AnchorRef,
            'stage2->stage3': stage3AnchorRef,
        };
        const anchorRef = scrollAnchors[`${prevPhase}->${phase}`];
        if (anchorRef) {
            requestAnimationFrame(() => {
                anchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
            return;
        }

        const isNearBottom =
            container.scrollHeight - container.scrollTop - container.clientHeight < 150;

        if (isNearBottom) {
            messagesEndRef.current?.scrollIntoView({ behavior: isLoading ? 'auto' : 'smooth' });
        }
    }, [conversation]);

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (searchPopoverRef.current && !searchPopoverRef.current.contains(e.target)) {
                setSearchPopoverOpen(false);
            }
        };
        if (searchPopoverOpen) document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [searchPopoverOpen]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !isLoading) {
            onSendMessage(input, activeSearchProvider);
            setInput('');
        }
    };

    const handleKeyDown = (e) => {
        // Submit on Enter (without Shift)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    if (!conversation) {
        if (mode === 'advisors') {
            return (
                <div className="chat-interface advisor-mode">
                    <div className="advisor-setup-scroll">
                        <AdvisorSetup
                            onStartDebate={onStartDebate}
                            isLoading={isLoading}
                        />
                    </div>
                </div>
            );
        }
        return (
            <div className="chat-interface">
                <div className="empty-state">
                    <h1>Welcome to LLM Council <span className="plus-text">Plus</span></h1>
                    <p className="hero-message">
                        Configure your council below, then start a session or ask your question.
                    </p>
                    <div className="welcome-grid-container">
                        <CouncilSetup
                            councilModels={councilModels}
                            chairmanModel={chairmanModel}
                            executionMode={executionMode}
                            editable
                            onCouncilChange={onCouncilChange}
                            onOpenSettings={onOpenSettings}
                        />
                    </div>
                    <button className="start-session-btn start-session-btn--secondary" onClick={onNewConversation}>
                        <span className="btn-content">
                            <span className="btn-icon">✨</span>
                            Start a New Council Session
                        </span>
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="chat-interface">
            {/* Messages Area */}
            <div className="messages-area" ref={messagesContainerRef}>
                {mode === 'advisors' && conversation.messages.length === 0 ? (
                    <div className="advisor-setup-scroll">
                        <AdvisorSetup
                            onStartDebate={onStartDebate}
                            isLoading={isLoading}
                        />
                    </div>
                ) : (conversation.messages.length === 0) ? (
                    <div className="hero-container">
                        <div className="hero-content">
                            <h1>Welcome to LLM Council <span className="text-gradient">Plus</span></h1>
                            <p className="hero-subtitle">
                                Configure your council below, then ask your question.
                            </p>
                            <div className="welcome-grid-container">
                                <CouncilSetup
                                    councilModels={councilModels}
                                    chairmanModel={chairmanModel}
                                    executionMode={executionMode}
                                    editable
                                    onCouncilChange={onCouncilChange}
                                    onOpenSettings={onOpenSettings}
                                />
                            </div>
                        </div>
                    </div>
                ) : (
                    conversation.messages.map((msg, index) => {
                        const isActiveCouncilTurn = msg.role === 'assistant'
                            && index === conversation.messages.length - 1
                            && isLoading;

                        return (
                        <div key={`${conversation.id}-msg-${index}`} className={`message ${msg.role}`}>
                            <div className="message-role">
                                {msg.role === 'user'
                                    ? (mode === 'advisors' ? 'Your Question' : 'Your Question to the Council')
                                    : (mode === 'advisors' ? 'Advisor Panel' : 'LLM Council')}
                            </div>

                            <div className="message-content">
                                {msg.role === 'user' ? (
                                    <MarkdownContent>{msg.content}</MarkdownContent>
                                ) : (msg.mode === 'advisors' || msg.type === 'advisor_debate') ? (
                                    <DebateView
                                        personas={msg.personas || []}
                                        rounds={msg.rounds || []}
                                        verdict={msg.verdict || null}
                                        tiebreaker={msg.tiebreaker || null}
                                        currentRound={msg.currentRound || msg.rounds?.length || 1}
                                        maxRounds={msg.maxRounds || msg.metadata?.max_rounds || 3}
                                        isRunning={msg.isRunning || false}
                                        question={msg.question || ''}
                                        webSearch={msg.webSearch}
                                        error={msg.error || null}
                                    />
                                ) : (
                                    <>
                                        {msg.error && (
                                            <div className="council-error">
                                                <span className="council-error-icon">⚠️</span>
                                                <span className="council-error-text">{msg.error}</span>
                                            </div>
                                        )}

                                        {isCouncilTurnPending(msg, isActiveCouncilTurn, isLoading) && (
                                            <div className="stage-loading">
                                                <div className="spinner"></div>
                                                <span>Consulting the council…</span>
                                            </div>
                                        )}

                                        {/* Search Loading */}
                                        {msg.loading?.search && (
                                            <div className="stage-loading">
                                                <div className="spinner"></div>
                                                <span>
                                                    🔍 Searching the web with {availableSearchProviders.find(p => p.id === (activeSearchProvider || searchProvider))?.name || 'Web'}...
                                                </span>
                                            </div>
                                        )}

                                        {/* Search Context */}
                                        {msg.metadata?.search_context && (
                                            <SearchContext
                                                searchQuery={msg.metadata?.search_query}
                                                extractedQuery={msg.metadata?.extracted_query}
                                                searchContext={msg.metadata?.search_context}
                                            />
                                        )}

                                        {/* Stage 1: Council Grid Visualization (during deliberation only) */}
                                        {shouldShowStage1CouncilGrid(msg) && (
                                            <div className="stage-container">
                                                <div className="stage-header">
                                                    <h3>Stage 1: Council Deliberation</h3>
                                                    {msg.timers?.stage1Start && (
                                                        <StageTimer
                                                            startTime={msg.timers.stage1Start}
                                                            endTime={msg.timers.stage1End}
                                                        />
                                                    )}
                                                </div>
                                                <CouncilGrid
                                                    models={councilModels}
                                                    chairman={chairmanModel}
                                                    status={msg.loading?.stage1 ? 'thinking' : 'complete'}
                                                    progress={{
                                                        currentModel: msg.progress?.stage1?.currentModel,
                                                        completed: msg.stage1?.map(r => r.model) || []
                                                    }}
                                                    showChairman={(msg.metadata?.execution_mode || executionMode) === 'full'}
                                                />
                                            </div>
                                        )}

                                        {renderStage1Content(msg)}

                                        {/* Stage 2 */}
                                        <div
                                            ref={isActiveCouncilTurn ? stage2AnchorRef : null}
                                            className="stage-scroll-anchor"
                                        >
                                            {msg.loading?.stage2 && <Stage2Skeleton />}
                                            {hasStage2Results(msg) && (
                                                <Stage2
                                                    rankings={msg.stage2}
                                                    labelToModel={msg.metadata?.label_to_model}
                                                    aggregateRankings={msg.metadata?.aggregate_rankings}
                                                    startTime={msg.timers?.stage2Start}
                                                    endTime={msg.timers?.stage2End}
                                                />
                                            )}
                                        </div>

                                        {/* Stage 3 */}
                                        <div
                                            ref={isActiveCouncilTurn ? stage3AnchorRef : null}
                                            className="stage-scroll-anchor"
                                        >
                                            {msg.loading?.stage3 && <Stage3Skeleton />}
                                            {msg.stage3 && (
                                                <Stage3
                                                    finalResponse={msg.stage3}
                                                    startTime={msg.timers?.stage3Start}
                                                    endTime={msg.timers?.stage3End}
                                                />
                                            )}
                                        </div>

                                        {/* Aborted Indicator */}
                                        {msg.aborted && (
                                            <div className="aborted-indicator">
                                                <span className="aborted-icon">⏹</span>
                                                <span className="aborted-text">
                                                    Generation stopped by user.
                                                    {hasStage1Results(msg) && !msg.stage3 && ' Partial results shown above.'}
                                                </span>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                        );
                    })
                )}

                {/* Bottom Spacer for floating input */}
                <div ref={messagesEndRef} style={{ height: '20px' }} />
            </div>

            {/* Floating Command Capsule — hidden for advisor debates */}
            {mode !== 'advisors' && <div className="input-area">
                {!councilConfigured ? (
                    <div className="input-container config-required">
                        <span className="config-message">
                            ⚠️ Council not ready — add at least one member
                            {executionMode === 'full' ? ' and a chairman' : ''}.
                            <button className="config-link" onClick={() => onOpenSettings('llm_keys')}>Configure API Keys</button>
                        </span>
                    </div>
                ) : (
                    <form className="input-container" onSubmit={handleSubmit}>
                        <div className="input-row-top">
                            <div className="search-provider-picker" ref={searchPopoverRef}>
                                <button
                                    type="button"
                                    className={`search-toggle ${activeSearchProvider ? 'active' : ''}`}
                                    onClick={() => !isLoading && setSearchPopoverOpen((v) => !v)}
                                    disabled={isLoading}
                                    title={activeSearchProvider ? `Search: ${availableSearchProviders.find(p => p.id === activeSearchProvider)?.name || activeSearchProvider}` : 'Web Search Off'}
                                    aria-haspopup="listbox"
                                    aria-expanded={searchPopoverOpen}
                                >
                                    <span className="search-icon">🌐</span>
                                    {activeSearchProvider && (
                                        <span className="search-label">
                                            {availableSearchProviders.find(p => p.id === activeSearchProvider)?.name || activeSearchProvider}
                                        </span>
                                    )}
                                </button>
                                {searchPopoverOpen && (
                                    <div className="search-popover" role="listbox">
                                        <button
                                            type="button"
                                            className={`search-popover-option ${!activeSearchProvider ? 'search-popover-option--selected' : ''}`}
                                            onClick={() => { setActiveSearchProvider(null); setSearchPopoverOpen(false); }}
                                        >
                                            <span className="search-popover-option-icon">✕</span>
                                            Off
                                        </button>
                                        {availableSearchProviders.map((p) => (
                                            <button
                                                key={p.id}
                                                type="button"
                                                className={`search-popover-option ${activeSearchProvider === p.id ? 'search-popover-option--selected' : ''}`}
                                                onClick={() => { setActiveSearchProvider(p.id); setSearchPopoverOpen(false); }}
                                            >
                                                <span className="search-popover-option-icon">🌐</span>
                                                {p.name}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <textarea
                                className="message-input"
                                placeholder={isLoading ? "Consulting..." : "Ask the Council..."}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                disabled={isLoading}
                                rows={1}
                                style={{ height: 'auto', minHeight: '24px' }}
                            />

                            {isLoading ? (
                                <button type="button" className="send-button stop-button" onClick={onAbort} title="Stop Generation">
                                    ⏹
                                </button>
                            ) : (
                                <button type="submit" className="send-button" disabled={!input.trim()}>
                                    ➤
                                </button>
                            )}
                        </div>

                        <div className="input-row-bottom">
                            <ExecutionModeToggle
                                value={executionMode}
                                onChange={onExecutionModeChange}
                                disabled={isLoading}
                            />
                        </div>
                    </form>
                )}
            </div>}
        </div>
    );
}
