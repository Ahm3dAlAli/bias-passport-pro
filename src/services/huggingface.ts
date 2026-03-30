const HF_API = 'https://huggingface.co/api/models';

export interface HFModel {
  modelId: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  pipeline_tag?: string;
  library_name?: string;
}

export async function searchHFModels(query: string, limit = 10): Promise<HFModel[]> {
  if (!query || query.length < 2) return [];
  try {
    const res = await fetch(
      `${HF_API}?search=${encodeURIComponent(query)}&filter=image-text-to-text&sort=downloads&direction=-1&limit=${limit}`
    );
    if (!res.ok) {
      // Fallback: search without filter
      const res2 = await fetch(
        `${HF_API}?search=${encodeURIComponent(query)}&sort=downloads&direction=-1&limit=${limit}`
      );
      if (!res2.ok) return [];
      return res2.json();
    }
    return res.json();
  } catch {
    return [];
  }
}

export async function getModelInfo(modelId: string): Promise<HFModel | null> {
  try {
    const res = await fetch(`${HF_API}/${modelId}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export function extractParamCount(tags: string[]): string {
  for (const tag of tags) {
    const m = tag.match(/^(\d+\.?\d*[BMK])$/i);
    if (m) return m[1].toUpperCase();
  }
  return 'Unknown';
}

export function extractProvider(modelId: string): string {
  const author = modelId.split('/')[0];
  const map: Record<string, string> = {
    'google': 'Google', 'Qwen': 'Alibaba', 'meta-llama': 'Meta',
    'OpenGVLab': 'OpenGVLab', 'HuggingFaceTB': 'HuggingFace',
    'vikhyatk': 'Moondream', 'openbmb': 'OpenBMB', 'microsoft': 'Microsoft',
    'mistralai': 'Mistral', 'deepseek-ai': 'DeepSeek',
  };
  return map[author] || author;
}
