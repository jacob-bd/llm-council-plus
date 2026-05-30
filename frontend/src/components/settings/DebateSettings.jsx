import React from 'react';

export default function DebateSettings({
    critiqueMode,
    setCritiqueMode,
    debateRounds,
    setDebateRounds,
    autoConverge,
    setAutoConverge,
    convergenceThreshold,
    setConvergenceThreshold,
    executionMode,
}) {
    return (
        <section>
            <h3>Debate Config</h3>
            <p className="section-description">
                Configure the multi-round iterative debate pipeline, convergence parameters, and critique formatting.
            </p>

            <div className="settings-group">
                <div className="setting-row">
                    <label>Critique Mode</label>
                    <div className="radio-group" style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                            <input type="radio" name="critiqueMode" value="freeform"
                                checked={critiqueMode === 'freeform'}
                                onChange={(e) => setCritiqueMode(e.target.value)} />
                            Free-form
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                            <input type="radio" name="critiqueMode" value="paragraph"
                                checked={critiqueMode === 'paragraph'}
                                onChange={(e) => setCritiqueMode(e.target.value)} />
                            Paragraph-level
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
                            <input type="radio" name="critiqueMode" value="claim"
                                checked={critiqueMode === 'claim'}
                                onChange={(e) => setCritiqueMode(e.target.value)} />
                            Claim-level
                        </label>
                    </div>
                </div>
                {critiqueMode !== 'freeform' && (
                    <p className="setting-hint">
                        {critiqueMode === 'claim'
                            ? 'Claim-level extracts canonical claims and maps evaluator verdicts directly over them. This adds ~1 extra API call per round for extraction.'
                            : 'Paragraph-level pre-numbers response paragraphs for stable, structured evaluation.'}
                    </p>
                )}
                <div className="setting-row" style={{ marginTop: '20px' }}>
                    <label>Number of Rounds</label>
                    <select value={debateRounds} onChange={(e) => setDebateRounds(Number(e.target.value))}>
                        {[1, 2, 3, 4, 5].map((n) => (
                            <option key={n} value={n}>{n}{n === 1 ? ' (single pass)' : ` rounds`}</option>
                        ))}
                    </select>
                </div>
                {debateRounds > 1 && (
                    <>
                        <div className="setting-row" style={{ marginTop: '20px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                                <input type="checkbox" checked={autoConverge} onChange={(e) => setAutoConverge(e.target.checked)} />
                                Auto-converge (stop early if rankings stabilize)
                            </label>
                        </div>
                        {autoConverge && (
                            <div className="setting-row" style={{ marginTop: '16px' }}>
                                <label>Convergence threshold</label>
                                <select value={convergenceThreshold} onChange={(e) => setConvergenceThreshold(Number(e.target.value))}>
                                    {[1, 2, 3].map((n) => (
                                        <option key={n} value={n}>{n} stable round{n > 1 ? 's' : ''}</option>
                                    ))}
                                </select>
                            </div>
                        )}
                        {executionMode === 'chat_only' && (
                            <p className="setting-hint" style={{ color: '#f59e0b', marginTop: '12px' }}>
                                ⚠️ Multi-round debate is not available in Chat Only mode.
                            </p>
                        )}
                        <p className="setting-hint" style={{ marginTop: '12px' }}>
                            More rounds = deeper analysis, but higher API consumption.
                        </p>
                    </>
                )}
            </div>
        </section>
    );
}
