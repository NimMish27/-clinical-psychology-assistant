import { FRAMEWORK_CARDS, type FrameworkCardData, type FrameworkCardId } from "./frameworks";

function norm(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function shared(a: string[], b: string[]): string[] {
  const bn = new Set(b.map(norm));
  return a.filter((x) => bn.has(norm(x)));
}

function onlyIn(a: string[], b: string[]): string[] {
  const bn = new Set(b.map(norm));
  return a.filter((x) => !bn.has(norm(x)));
}

export interface FrameworkComparison {
  a: FrameworkCardData;
  b: FrameworkCardData;
  focusShared: boolean;
  confidenceClose: boolean;
  sharedTags: string[];
  onlyATags: string[];
  onlyBTags: string[];
  sharedStrengths: string[];
  onlyAStrengths: string[];
  onlyBStrengths: string[];
  sharedLimitations: string[];
  onlyALimitations: string[];
  onlyBLimitations: string[];
  sharedInterventions: string[];
  onlyAInterventions: string[];
  onlyBInterventions: string[];
  sharedCount: number;
  uniqueCount: number;
  overlapPct: number;
}

export function compareFrameworks(
  aId: FrameworkCardId,
  bId: FrameworkCardId,
): FrameworkComparison {
  const a = FRAMEWORK_CARDS.find((c) => c.id === aId)!;
  const b = FRAMEWORK_CARDS.find((c) => c.id === bId)!;

  const focusShared = norm(a.focus) === norm(b.focus);
  const confidenceClose = Math.abs(a.confidence - b.confidence) < 0.1;

  const sharedTags = shared(a.tags, b.tags);
  const onlyATags = onlyIn(a.tags, b.tags);
  const onlyBTags = onlyIn(b.tags, a.tags);

  const aInterventions = a.workspace.suggestedInterventions.map((i) => i.name);
  const bInterventions = b.workspace.suggestedInterventions.map((i) => i.name);
  const sharedInterventions = shared(aInterventions, bInterventions);
  const onlyAInterventions = onlyIn(aInterventions, bInterventions);
  const onlyBInterventions = onlyIn(bInterventions, aInterventions);

  const sharedStrengths = shared(a.workspace.strengths, b.workspace.strengths);
  const onlyAStrengths = onlyIn(a.workspace.strengths, b.workspace.strengths);
  const onlyBStrengths = onlyIn(b.workspace.strengths, a.workspace.strengths);

  const sharedLimitations = shared(a.workspace.limitations, b.workspace.limitations);
  const onlyALimitations = onlyIn(a.workspace.limitations, b.workspace.limitations);
  const onlyBLimitations = onlyIn(b.workspace.limitations, a.workspace.limitations);

  const sharedCount =
    sharedTags.length +
    sharedStrengths.length +
    sharedLimitations.length +
    sharedInterventions.length;
  const uniqueCount =
    onlyATags.length +
    onlyBTags.length +
    onlyAStrengths.length +
    onlyBStrengths.length +
    onlyALimitations.length +
    onlyBLimitations.length +
    onlyAInterventions.length +
    onlyBInterventions.length;

  const total = sharedCount + uniqueCount;

  return {
    a,
    b,
    focusShared,
    confidenceClose,
    sharedTags,
    onlyATags,
    onlyBTags,
    sharedStrengths,
    onlyAStrengths,
    onlyBStrengths,
    sharedLimitations,
    onlyALimitations,
    onlyBLimitations,
    sharedInterventions,
    onlyAInterventions,
    onlyBInterventions,
    sharedCount,
    uniqueCount,
    overlapPct: total === 0 ? 0 : Math.round((sharedCount / total) * 100),
  };
}

export type ItemKind = "shared" | "only";
