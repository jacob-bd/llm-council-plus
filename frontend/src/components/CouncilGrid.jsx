import React from 'react';
import { createPortal } from 'react-dom';
import './CouncilGrid.css';

// Import Provider Icons
import openaiLogo from '../assets/icons/openai.svg';
import anthropicLogo from '../assets/icons/anthropic.svg';
import googleLogo from '../assets/icons/google.svg';
import mistralLogo from '../assets/icons/mistral.svg';
import ollamaLogo from '../assets/icons/ollama.svg';
import deepseekLogo from '../assets/icons/deepseek.svg';
import groqLogo from '../assets/icons/groq.svg';
import openrouterLogo from '../assets/icons/openrouter.svg';
import customLogo from '../assets/icons/openai-compatible.svg';

const PROVIDER_CONFIG = {
    openai: { color: '#10a37f', label: 'OpenAI', logo: openaiLogo },
    anthropic: { color: '#d97757', label: 'Anthropic', logo: anthropicLogo },
    google: { color: '#4285f4', label: 'Google', logo: googleLogo },
    mistral: { color: '#fcd34d', label: 'Mistral', logo: mistralLogo },
    groq: { color: '#f55036', label: 'Groq', logo: groqLogo },
    ollama: { color: '#ffffff', label: 'Local', logo: ollamaLogo },
    deepseek: { color: '#4e61e6', label: 'DeepSeek', logo: deepseekLogo },
    openrouter: { color: '#7f5af0', label: 'OpenRouter', logo: openrouterLogo },
    custom: { color: '#06b6d4', label: 'Custom', logo: customLogo },
    default: { color: '#888888', label: 'Model', logo: null, icon: '🤖' }
};

const getProviderInfo = (modelId) => {
    if (!modelId) return PROVIDER_CONFIG.default;
    const id = modelId.toLowerCase();

    // Check for provider prefixes FIRST (order matters!)
    if (id.startsWith('custom:')) return PROVIDER_CONFIG.custom;
    if (id.startsWith('ollama:')) return PROVIDER_CONFIG.ollama;
    if (id.startsWith('groq:')) return PROVIDER_CONFIG.groq;

    // OpenRouter handling
    if (id.startsWith('openrouter:') || id.includes('openrouter')) return PROVIDER_CONFIG.openrouter;

    // Check for OpenRouter path format (provider/model)
    // This ensures ALL OpenRouter models get the OpenRouter icon if they follow the standard format
    if (id.includes('/')) return PROVIDER_CONFIG.openrouter;

    // Check for specific model identifiers (only if no prefix matched)
    if (id.includes('gpt') || id.includes('openai')) return PROVIDER_CONFIG.openai;
    if (id.includes('claude') || id.includes('anthropic')) return PROVIDER_CONFIG.anthropic;
    if (id.includes('gemini') || id.includes('google')) return PROVIDER_CONFIG.google;
    if (id.includes('mistral') || id.includes('mixtral')) return PROVIDER_CONFIG.mistral;
    if (id.includes('deepseek')) return PROVIDER_CONFIG.deepseek;

    // Fallback for other known patterns
    if (id.includes('llama') || id.includes('grok')) {
        return PROVIDER_CONFIG.openrouter;
    }

    return PROVIDER_CONFIG.default;
};

const getModelDisplayName = (modelId) => {
    if (!modelId) return 'Model';
    if (modelId.startsWith('placeholder')) return 'Council Member';

    let name = modelId;

    // Remove :free suffix first (from OpenRouter free models)
    name = name.replace(/:free$/, '');

    // Remove provider prefixes (e.g., "openrouter:", "ollama:", "groq:")
    if (name.includes(':')) {
        name = name.split(':').slice(1).join(':');
    }

    // Remove path-based prefixes (e.g., "openai/", "anthropic/")
    if (name.includes('/')) {
        name = name.split('/').pop();
    }

    return name;
};

export default function CouncilGrid({
    models = [],
    chairman = null,
    status = 'idle', // 'idle', 'thinking', 'complete'
    progress = {}    // { currentModel: 'id', completed: ['id1', 'id2'] }
}) {
    // Filter out empty/null model IDs, then use placeholders if none remain
    const validModels = models.filter(m => m && m.trim() !== '');
    const displayModels = validModels.length > 0 ? validModels : ['placeholder-1', 'placeholder-2', 'placeholder-3'];


    // Debug: Log model IDs


    // Tooltip State
    const [tooltip, setTooltip] = React.useState({ visible: false, x: 0, y: 0, content: '' });

    const handleMouseEnter = (e, modelId) => {
        const content = getModelDisplayName(modelId);
        setTooltip({
            visible: true,
            x: e.clientX,
            y: e.clientY,
            content
        });
    };

    const handleMouseMove = (e) => {
        setTooltip(prev => ({
            ...prev,
            x: e.clientX,
            y: e.clientY
        }));
    };

    const handleMouseLeave = () => {
        setTooltip(prev => ({ ...prev, visible: false }));
    };

    // Helper to get chairman info
    const chairmanInfo = chairman ? getProviderInfo(chairman) : null;

    // Calculate grid layout based on member count
    const memberCount = displayModels.length;
    let gridClass = 'council-grid';

    if (memberCount <= 2) {
        gridClass += ' layout-2-members';
    } else if (memberCount === 3) {
        gridClass += ' layout-3-members';
    } else if (memberCount === 4) {
        gridClass += ' layout-4-members';
    } else if (memberCount === 5) {
        gridClass += ' layout-5-members';
    } else if (memberCount === 6) {
        gridClass += ' layout-6-members';
    } else if (memberCount === 7) {
        gridClass += ' layout-7-members';
    } else {
        gridClass += ' layout-8-members'; // 8 or more
    }

    return (
        <div className={gridClass}>
            {/* Tooltip Portal */}
            {tooltip.visible && createPortal(
                <div
                    className="custom-tooltip"
                    style={{ left: tooltip.x, top: tooltip.y }}
                >
                    {tooltip.content}
                </div>,
                document.body
            )}

            {/* Regular Council Members */}
            {displayModels.map((modelId, index) => {
                const isPlaceholder = modelId.startsWith('placeholder');
                const info = isPlaceholder ? PROVIDER_CONFIG.default : getProviderInfo(modelId);
                const displayName = getModelDisplayName(modelId);

                // Determine state
                let cardState = 'idle';
                if (status === 'thinking') {
                    if (progress.completed?.includes(modelId)) {
                        cardState = 'done';
                    } else if (progress.currentModel === modelId) {
                        cardState = 'active';
                    } else {
                        cardState = 'waiting';
                    }
                } else if (status === 'complete') {
                    cardState = 'done';
                } else if (status === 'idle') {
                    cardState = 'ready';
                }

                return (
                    <div
                        key={index}
                        className={`council-card ${cardState}`}
                        style={{ '--provider-color': info.color }}
                        onMouseEnter={(e) => handleMouseEnter(e, modelId)}
                        onMouseMove={handleMouseMove}
                        onMouseLeave={handleMouseLeave}
                    >
                        <div className="role-badge member">Member #{index + 1}</div>
                        <div className="council-avatar">
                            {info.logo ? (
                                <img src={info.logo} alt={info.label} className="provider-logo" />
                            ) : (
                                <span className="avatar-icon">{info.icon}</span>
                            )}
                            {cardState === 'active' && <div className="thinking-ring"></div>}
                            {cardState === 'done' && <div className="done-badge">✓</div>}
                        </div>
                        <div className="council-info">
                            <span className="model-name">
                                {displayName}
                            </span>
                            <span className="provider-label">{info.label}</span>
                        </div>
                    </div>
                );
            })}

            {/* Chairman Card - Always show, but state changes */}
            <div
                className={`council-card chairman ${status === 'thinking' ? 'waiting' : 'ready'}`}
                style={{ '--provider-color': (status !== 'thinking' && chairman) ? getProviderInfo(chairman).color : '#94a3b8' }}
                onMouseEnter={(e) => status !== 'thinking' && handleMouseEnter(e, chairman || 'Chairman')}
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
            >
                <div className="role-badge chairman">Chairman</div>
                <div className="council-avatar">
                    {status !== 'thinking' && chairmanInfo && chairmanInfo.logo ? (
                        <img
                            src={chairmanInfo.logo}
                            alt={chairmanInfo.label}
                            className="provider-logo"
                        />
                    ) : (
                        <span className="avatar-icon">{status === 'thinking' ? '⏳' : (chairmanInfo ? chairmanInfo.icon : '⚖️')}</span>
                    )}
                </div>
                <div className="council-info">
                    <span className="model-name">
                        {status === 'thinking' ? 'Verdict Pending' : (chairman ? getModelDisplayName(chairman) : 'Model')}
                    </span>
                    <span className="provider-label">
                        {status === 'thinking' ? 'Waiting...' : 'Final Verdict'}
                    </span>
                </div>
            </div>
        </div>
    );
}
