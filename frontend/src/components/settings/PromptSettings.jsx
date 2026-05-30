const COUNCIL_PROMPTS = [
  {
    id: 'stage1',
    key: 'stage1_prompt',
    label: 'Stage 1',
    heading: 'Stage 1: Initial Response',
    description: 'Used when each council model answers the user question independently.',
    variables: ['user_query', 'search_context_block'],
  },
  {
    id: 'stage2',
    key: 'stage2_prompt',
    label: 'Stage 2',
    heading: 'Stage 2: Peer Ranking',
    description: 'Used when council models anonymously review and rank the Stage 1 responses.',
    variables: ['user_query', 'responses_text', 'search_context_block'],
  },
  {
    id: 'stage3',
    key: 'stage3_prompt',
    label: 'Stage 3',
    heading: 'Stage 3: Chairman Synthesis',
    description: 'Used when the chairman model synthesizes the council output into the final answer.',
    variables: ['user_query', 'stage1_responses', 'rankings_text', 'search_context_block'],
  },
  {
    id: 'stage4',
    key: 'stage4_prompt',
    label: 'Stage 4',
    heading: 'Stage 4: Corrected Draft (Multi-Round)',
    description: 'Used when the chairman synthesizes claims and edits the previous responses into a finalized, revised draft after multiple debate rounds.',
    variables: ['total_rounds', 'original_text', 'verdict_text'],
  },
  {
    id: 'title',
    key: 'title_prompt',
    label: 'Title',
    heading: 'Conversation Title',
    description: 'Used to generate a short title for saved conversations.',
    variables: ['user_query'],
  },
  {
    id: 'query',
    key: 'query_prompt',
    label: 'Query',
    heading: 'Search Query Generation',
    description: 'Used to turn a user question into a concise web search query.',
    variables: ['user_query'],
  },
];

const ADVISOR_PROMPTS = [
  {
    id: 'advisor_round1',
    key: 'advisor_round1_prompt',
    label: 'Round 1',
    heading: 'Round 1: Opening Position',
    description: 'Used for each advisor opening statement before the debate has peer claims to rebut.',
    variables: ['search_context_block', 'question', 'consensus_tag'],
  },
  {
    id: 'advisor_followup',
    key: 'advisor_followup_prompt',
    label: 'Follow-up',
    heading: 'Round 2+: Structured Critique',
    description: 'Used after Round 1 so advisors respond to the distilled strongest claims from peers.',
    variables: [
      'search_context_block',
      'question',
      'transcript',
      'round_number',
      'previous_round_number',
      'cross_pollination_extract',
      'consensus_tag',
    ],
  },
  {
    id: 'advisor_extract',
    key: 'advisor_cross_pollination_prompt',
    label: 'Extract',
    heading: 'Cross-Pollination Extract',
    description: 'Used by the tiebreaker model after each round to summarize the strongest claims.',
    variables: ['question', 'round_number', 'round_transcript'],
  },
  {
    id: 'advisor_verdict',
    key: 'advisor_verdict_prompt',
    label: 'Verdict',
    heading: 'Final Advisor Verdict',
    description: 'Used to synthesize the full advisor debate and the debate arc into a final verdict.',
    variables: ['question', 'transcript', 'debate_arc'],
  },
  {
    id: 'advisor_tiebreaker',
    key: 'advisor_tiebreaker_prompt',
    label: 'Tiebreaker',
    heading: 'Two-Advisor Tiebreaker',
    description: 'Used when two advisors fail to reach consensus and a tiebreaker verdict is needed.',
    variables: ['question', 'transcript'],
  },
];

export default function PromptSettings({
  variant = 'council',
  prompts,
  handlePromptChange,
  handleResetPrompt,
  activePromptTab,
  setActivePromptTab,
  stage2Temperature,
  setStage2Temperature,
}) {
  const promptConfigs = variant === 'advisor' ? ADVISOR_PROMPTS : COUNCIL_PROMPTS;
  const activeConfig = promptConfigs.find(prompt => prompt.id === activePromptTab) || promptConfigs[0];
  const sectionTitle = variant === 'advisor' ? 'Advisor System Prompts' : 'Council System Prompts';
  const sectionDescription = variant === 'advisor'
    ? 'Customize the orchestration prompts used by advisor debates.'
    : 'Customize the system prompts used by the council workflow.';
  const textAreaRows = variant === 'advisor' ? 18 : 20;

  return (
    <section>
      <h3>{sectionTitle}</h3>
      <p className="section-description">{sectionDescription}</p>

      <div className="prompts-tabs">
        {promptConfigs.map(prompt => (
          <button
            key={prompt.id}
            type="button"
            className={`prompt-tab ${activeConfig.id === prompt.id ? 'active' : ''}`}
            onClick={() => setActivePromptTab(prompt.id)}
          >
            {prompt.label}
          </button>
        ))}
      </div>

      <div className="prompt-content">
        <label htmlFor={`${activeConfig.id}-prompt`}>{activeConfig.heading}</label>
        <p className="prompt-help">
          {activeConfig.description}
          <span>Available variables:</span>
          {activeConfig.variables.map(variable => (
            <code key={variable}>{`{${variable}}`}</code>
          ))}
        </p>
        <textarea
          id={`${activeConfig.id}-prompt`}
          value={prompts[activeConfig.key] || ''}
          onChange={(event) => handlePromptChange(activeConfig.key, event.target.value)}
          rows={textAreaRows}
          spellCheck={false}
        />
        <button
          type="button"
          className="reset-prompt-btn"
          onClick={() => handleResetPrompt(activeConfig.key)}
        >
          Reset to Default
        </button>
      </div>

      {variant === 'council' && activeConfig.id === 'stage2' && setStage2Temperature && (
        <div className="config-section" style={{ marginTop: '24px' }}>
          <label className="config-label">Stage 2 Heat</label>
          <p className="section-description">
            Controls how much variation the peer ranking step allows. Lower values keep ranking more consistent.
          </p>
          <div className="temperature-control">
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={stage2Temperature}
              onChange={(event) => setStage2Temperature(parseFloat(event.target.value))}
            />
            <span className="temperature-value">{stage2Temperature}</span>
          </div>
        </div>
      )}
    </section>
  );
}
