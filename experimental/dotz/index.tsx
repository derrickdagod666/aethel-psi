/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Part, Content, GoogleGenAI as GoogleGenAIType, FunctionCall, Type } from '@google/genai';
import { marked } from 'marked';

// --- DOM Elements ---
const chatHistoryEl = document.getElementById('chat-history')!;
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement;
const sendButton = document.getElementById('send-button') as HTMLButtonElement;
const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
const imagePreviewContainer = document.getElementById('image-preview-container')!;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const removeImageButton = document.getElementById('remove-image') as HTMLButtonElement;
const ttsToggle = document.getElementById('tts-toggle') as HTMLButtonElement;
const useGoogleSearchCheckbox = document.getElementById('use-google-search') as HTMLInputElement;


// --- Constants ---
const TTS_STORAGE_KEY = 'gemini-chat-tts-enabled';

// --- State ---
let attachedImage: { base64: string; mimeType: string; } | null = null;
let history: Content[] = [];
let isProcessing = false;
let isTtsEnabled = true; // Default to on
let speechBuffer = '';
let preferredVoice: SpeechSynthesisVoice | null = null;
let hasInteracted = false;
let ttsKeepAliveInterval: number | null = null; // For TTS stability

// --- Helper Functions ---

/**
 * Converts a File object to a base64 string.
 */
function fileToGenerativePart(file: File): Promise<{ base64: string; mimeType: string; }> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      const base64 = result.split(',')[1];
      resolve({ base64, mimeType: file.type });
    };
    reader.readAsDataURL(file);
  });
}

/**
 * Adds a message to the chat history UI.
 */
function addMessage(role: 'user' | 'model', text: string, imageUrl?: string): HTMLElement {
  const messageEl = document.createElement('div');
  messageEl.className = `message ${role}-message`;

  const contentEl = document.createElement('div');
  contentEl.className = 'content';

  // For model messages that will be streamed into, we need to create the
  // text container div up front. For user messages, text will be present.
  if (role === 'model' || text) {
    const textEl = document.createElement('div');
    if (text) {
        textEl.innerHTML = marked.parse(text, { async: false }) as string;
    }
    contentEl.appendChild(textEl);
  }

  if (imageUrl) {
    const imageEl = document.createElement('img');
    imageEl.src = imageUrl;
    imageEl.alt = 'User-uploaded image';
    contentEl.appendChild(imageEl);
  }

  messageEl.appendChild(contentEl);
  chatHistoryEl.appendChild(messageEl);
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

  return messageEl;
}

/**
 * Adds a message with a generated image to the chat history.
 */
function addModelImageMessage(imageUrl: string, altText: string): HTMLElement {
    const messageEl = document.createElement('div');
    messageEl.className = 'message model-message';

    const contentEl = document.createElement('div');
    contentEl.className = 'content';

    const imageEl = document.createElement('img');
    imageEl.src = imageUrl;
    imageEl.alt = altText;
    contentEl.appendChild(imageEl);
    
    messageEl.appendChild(contentEl);
    chatHistoryEl.appendChild(messageEl);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

    return messageEl;
}


/**
 * Renders a structured recipe object into a styled card in the chat history.
 */
function addRecipeCard(recipe: any): HTMLElement {
  const messageEl = document.createElement('div');
  messageEl.className = 'message model-message';

  const contentEl = document.createElement('div');
  contentEl.className = 'content recipe-card';

  const title = document.createElement('h2');
  title.textContent = recipe.recipeName;
  contentEl.appendChild(title);

  if (recipe.description) {
    const description = document.createElement('p');
    description.className = 'recipe-description';
    description.textContent = recipe.description;
    contentEl.appendChild(description);
  }

  const ingredientsTitle = document.createElement('h3');
  ingredientsTitle.textContent = 'Ingredients';
  contentEl.appendChild(ingredientsTitle);

  const ingredientsList = document.createElement('ul');
  recipe.ingredients.forEach((item: string) => {
    const li = document.createElement('li');
    li.textContent = item;
    ingredientsList.appendChild(li);
  });
  contentEl.appendChild(ingredientsList);

  const instructionsTitle = document.createElement('h3');
  instructionsTitle.textContent = 'Instructions';
  contentEl.appendChild(instructionsTitle);

  const instructionsList = document.createElement('ol');
  recipe.instructions.forEach((item: string) => {
    const li = document.createElement('li');
    li.textContent = item;
    instructionsList.appendChild(li);
  });
  contentEl.appendChild(instructionsList);

  messageEl.appendChild(contentEl);
  chatHistoryEl.appendChild(messageEl);
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

  return messageEl;
}


/**
 * Adds a visually distinct message for tool calls.
 */
function addToolCallMessage(text: string): HTMLElement {
    const messageEl = document.createElement('div');
    messageEl.className = 'message model-message tool-call-message';

    const contentEl = document.createElement('div');
    contentEl.className = 'content';
    contentEl.textContent = text;
    
    messageEl.appendChild(contentEl);
    chatHistoryEl.appendChild(messageEl);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

    return messageEl;
}


/**
 * Adds a loading indicator for the model's response.
 */
function addLoadingIndicator(): HTMLElement {
    const messageEl = document.createElement('div');
    messageEl.className = 'message model-message';

    const contentEl = document.createElement('div');
    contentEl.className = 'content loading-dots';
    contentEl.innerHTML = '<span></span><span></span><span></span>';

    messageEl.appendChild(contentEl);
    chatHistoryEl.appendChild(messageEl);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

    return messageEl;
}

/**
 * Toggles the UI state between busy and idle.
 */
function setBusyState(busy: boolean) {
  isProcessing = busy;
  promptInput.disabled = busy;
  sendButton.disabled = busy;
  imageUpload.disabled = busy;
  useGoogleSearchCheckbox.disabled = busy;
}

/**
 * Clears the image preview and associated state.
 */
function clearImagePreview() {
  attachedImage = null;
  imagePreview.src = '';
  imageUpload.value = '';
  imagePreviewContainer.style.display = 'none';
}

// --- UI Enhancement Functions ---

function addCodeCopyButtons(messageEl: HTMLElement) {
    const codeBlocks = messageEl.querySelectorAll('pre');
    codeBlocks.forEach(pre => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        pre.appendChild(button);

        button.addEventListener('click', () => {
            const code = pre.querySelector('code')?.innerText || '';
            navigator.clipboard.writeText(code).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
    });
}

function renderGroundingSources(sources: any[], messageEl: HTMLElement) {
    if (!sources || sources.length === 0) return;

    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'grounding-sources';
    
    const title = document.createElement('h3');
    title.textContent = 'Sources';
    sourcesContainer.appendChild(title);

    const list = document.createElement('ul');
    sources.forEach(source => {
        if (source.web) {
            const item = document.createElement('li');
            const link = document.createElement('a');
            link.href = source.web.uri;
            link.textContent = source.web.title || source.web.uri;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            item.appendChild(link);
            list.appendChild(item);
        }
    });

    sourcesContainer.appendChild(list);
    messageEl.querySelector('.content')?.appendChild(sourcesContainer);
}


// --- TTS Functions ---

/**
 * Selects the best available English speech synthesis voice.
 * This function scores and prioritizes voices to find the most natural-sounding
 * option, preferring high-quality, cloud-based voices.
 */
function getBestVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
    if (!voices || voices.length === 0) {
        return null;
    }

    const scoredVoices = voices
        .filter(v => v.lang.startsWith('en-')) // Only consider English voices
        .map(voice => {
            let score = 0;
            const name = voice.name.toLowerCase();
            const lang = voice.lang.toLowerCase();

            // Prioritize known high-quality voices (cloud-based often sound better)
            if (name.includes('google')) score += 10;
            if (name.includes('microsoft') && name.includes('online')) score += 10;
            if (name.includes('samantha') || name.includes('alex') || name.includes('daniel') || name.includes('fiona')) score += 8; // Good Apple voices
            if (name.includes('zira') || name.includes('david')) score += 8; // Good MS voices

            // General preferences
            if (!voice.localService) score += 5; // Prefer cloud voices for quality
            if (lang === 'en-us') score += 2;    // Prefer US English
            if (voice.default) score += 1;       // Give a slight bump to the default

            return { voice, score };
        })
        .sort((a, b) => b.score - a.score);

    if (scoredVoices.length > 0) {
        return scoredVoices[0].voice;
    }
    
    // Absolute fallback if no english voices were scored
    return voices.find(v => v.lang.startsWith('en-')) || voices[0] || null;
}

/**
 * Stops the TTS keep-alive interval.
 */
function stopTtsKeepAlive() {
    if (ttsKeepAliveInterval) {
        clearInterval(ttsKeepAliveInterval);
        ttsKeepAliveInterval = null;
    }
}

/**
 * Starts a keep-alive interval to prevent the speech synthesis engine
 * from going idle on some platforms.
 */
function startTtsKeepAlive() {
    stopTtsKeepAlive(); // Prevent duplicates
    if (!isTtsEnabled || !window.speechSynthesis) return;

    ttsKeepAliveInterval = window.setInterval(() => {
        // If nothing is happening, resume and send a silent utterance
        if (!window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
            window.speechSynthesis.resume(); 
            const utterance = new SpeechSynthesisUtterance('');
            utterance.volume = 0;
            window.speechSynthesis.speak(utterance);
        }
    }, 10000); // Every 10 seconds
}


function speak(text: string) {
    if (!isTtsEnabled || !text.trim() || !window.speechSynthesis) return;
    
    stopTtsKeepAlive(); // Stop keep-alive when we are about to speak for real

    // If the API is in a weird paused state, try to resume it
    if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
    }
    
    const utterance = new SpeechSynthesisUtterance(text);
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }
    
    utterance.onerror = (event) => {
        console.error('SpeechSynthesisUtterance.onerror', event);
        startTtsKeepAlive(); // Restart keep-alive on error to be safe
    };

    utterance.onend = () => {
        startTtsKeepAlive(); // Restart keep-alive when speech finishes
    };

    window.speechSynthesis.speak(utterance);
}

function processSpeech(textChunk: string, flush = false) {
    if (!isTtsEnabled) return;
    speechBuffer += textChunk;
    if (flush) {
        speak(speechBuffer);
        speechBuffer = '';
        return;
    }
    const sentenceEndRegex = /(?<=[.?!])\s+/;
    let sentences = speechBuffer.split(sentenceEndRegex);
    if (sentences.length > 1) {
        const completeSentences = sentences.slice(0, -1).join('');
        speak(completeSentences);
        speechBuffer = sentences[sentences.length - 1];
    }
}

function setupTts() {
    if (!window.speechSynthesis) {
        ttsToggle.style.display = 'none';
        return;
    }
    function updateTtsButton(enabled: boolean) {
        ttsToggle.setAttribute('aria-pressed', String(enabled));
        const label = enabled ? 'Disable text-to-speech' : 'Enable text-to-speech';
        ttsToggle.setAttribute('aria-label', label);
        ttsToggle.title = label;
    }
    function loadVoices() {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            preferredVoice = getBestVoice(voices);
        }
    }
    
    // Eagerly try to load voices
    loadVoices();
    // The 'onvoiceschanged' event is crucial for when voices load asynchronously
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    const storedSetting = localStorage.getItem(TTS_STORAGE_KEY);
    if (storedSetting !== null) {
        isTtsEnabled = storedSetting === 'true';
    }
    updateTtsButton(isTtsEnabled);

    ttsToggle.addEventListener('click', () => {
        isTtsEnabled = !isTtsEnabled;
        localStorage.setItem(TTS_STORAGE_KEY, String(isTtsEnabled));
        updateTtsButton(isTtsEnabled);
        if (!isTtsEnabled) {
            window.speechSynthesis.cancel(); // Stop any current speech
            speechBuffer = '';
            stopTtsKeepAlive(); // Stop keep-alive when TTS is disabled
        } else if (hasInteracted) {
            startTtsKeepAlive(); // Start it if TTS is re-enabled and user has interacted
        }
    });
}

function warmUpTts() {
    if (!hasInteracted && isTtsEnabled && window.speechSynthesis) {
        // Cancel any previous state from a page refresh
        window.speechSynthesis.cancel(); 
        // Speak a silent utterance to activate the audio context.
        const utterance = new SpeechSynthesisUtterance('');
        utterance.volume = 0;
        window.speechSynthesis.speak(utterance);
        // Start the keep-alive mechanism after the first interaction
        startTtsKeepAlive();
    }
    hasInteracted = true;
}

// --- Tool Definitions ---
const tools = [{
  functionDeclarations: [
    {
      name: 'get_weather',
      description: 'Get the current weather in a given location',
      parameters: {
        type: Type.OBJECT,
        properties: { location: { type: Type.STRING, description: 'The city and state, e.g. San Francisco, CA' } },
        required: ['location'],
      },
    },
    {
        name: 'get_stock_price',
        description: 'Get the current stock price for a given stock ticker symbol',
        parameters: {
            type: Type.OBJECT,
            properties: { symbol: { type: Type.STRING, description: 'The stock ticker symbol, e.g. GOOG' } },
            required: ['symbol'],
        },
    },
    {
        name: 'generate_image',
        description: 'Generates an image based on a user description. Use this when the user asks to draw, create, make, or generate a picture or image.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                prompt: {
                    type: Type.STRING,
                    description: 'A detailed description of the image to generate.'
                }
            },
            required: ['prompt'],
        },
    }
  ]
}];

const recipeSchema = {
  type: Type.OBJECT,
  properties: {
    recipeName: { type: Type.STRING, description: 'The name of the recipe.' },
    description: { type: Type.STRING, description: 'A short description of the recipe.' },
    ingredients: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: 'A list of ingredients with quantities.'
    },
    instructions: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: 'A list of step-by-step instructions.'
    }
  },
  required: ['recipeName', 'ingredients', 'instructions']
};

const systemInstructionText = 'You are a helpful and creative assistant. Your responses should be conversational, empathetic, and aim to fully address the user\'s query. When providing information, format it in a way that is clear and easy to read, and sounds natural when converted to speech. Be proactive in offering help and suggestions.';


// --- Main Application Logic ---

async function main() {
  setupTts();

  let ai: GoogleGenAIType;
  try {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  } catch (error) {
    console.error(error);
    addMessage('model', 'Error: Could not initialize the Gemini API. Please check the API key and reload the page.');
    setBusyState(true);
    return;
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isProcessing) return;

    if (!hasInteracted) warmUpTts();
    window.speechSynthesis.cancel();
    speechBuffer = '';

    const prompt = promptInput.value.trim();
    const image = attachedImage;
    const useGoogleSearch = useGoogleSearchCheckbox.checked;

    if (!prompt && !image) return;

    const wantsRecipe = prompt.toLowerCase().includes('recipe') && !useGoogleSearch;

    setBusyState(true);
    promptInput.value = '';
    // Keep the checkbox state for multi-turn conversations
    // useGoogleSearchCheckbox.checked = false; 

    const userMessageParts: Part[] = [];
    if (prompt) userMessageParts.push({ text: prompt });
    if (image) userMessageParts.push({ inlineData: { data: image.base64, mimeType: image.mimeType } });
    
    history.push({ role: 'user', parts: userMessageParts });
    addMessage('user', prompt, image ? `data:${image.mimeType};base64,${image.base64}` : undefined);
    clearImagePreview();
    
    let isDone = false;
    while (!isDone) {
        const loadingIndicator = addLoadingIndicator();
        try {
            const config: any = {};
            if (useGoogleSearch) {
                config.tools = [{ googleSearch: {} }];
            } else if (wantsRecipe) {
                config.responseMimeType = 'application/json';
                config.responseSchema = recipeSchema;
                config.systemInstruction = systemInstructionText;
            } else {
                config.tools = tools;
                config.systemInstruction = systemInstructionText;
            }

            const result = await ai.models.generateContentStream({
                model: 'gemini-2.5-flash',
                contents: history,
                config: config,
            });
            
            loadingIndicator.remove();
            
            let fullText = '';
            const functionCalls: FunctionCall[] = [];
            let modelResponseEl: HTMLElement | null = null;
            let groundingMetadata: any;
            
            for await (const chunk of result) {
                if (chunk.functionCalls) {
                    functionCalls.push(...chunk.functionCalls);
                }

                const chunkText = chunk.text;
                if (chunkText) {
                    fullText += chunkText;
                    if (!wantsRecipe) {
                        if (!modelResponseEl) {
                            modelResponseEl = addMessage('model', '');
                        }
                        const contentDiv = modelResponseEl.querySelector('.content div')!;
                        contentDiv.innerHTML = marked.parse(fullText, { async: false }) as string;
                        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
                        processSpeech(chunkText);
                    }
                }
                const chunkGroundingMetadata = chunk.candidates?.[0]?.groundingMetadata;
                if (chunkGroundingMetadata) {
                    groundingMetadata = chunkGroundingMetadata;
                }
            }
            
            if (functionCalls.length > 0) {
                 if (modelResponseEl) modelResponseEl.remove();

                 history.push({ role: 'model', parts: [{ functionCall: functionCalls[0] }] });
                 const call = functionCalls[0];

                 if (call.name === 'get_weather') {
                    const { location } = call.args;
                    addToolCallMessage(`Calling weather tool for ${location}...`);
                    const weatherResponse = {
                        location,
                        temperature: `${Math.floor(Math.random() * 25 + 5)}°C`,
                        conditions: ['Sunny', 'Cloudy', 'Rainy', 'Windy'][Math.floor(Math.random() * 4)],
                    };
                    history.push({
                        role: 'user',
                        parts: [{ functionResponse: { name: 'get_weather', response: weatherResponse } }]
                    });
                 } else if (call.name === 'get_stock_price') {
                    const { symbol } = call.args;
                    const upperSymbol = (symbol as string).toUpperCase();
                    addToolCallMessage(`Calling stock price tool for ${upperSymbol}...`);
                    const stockResponse = {
                        symbol: upperSymbol,
                        price: (Math.random() * 1000 + 100).toFixed(2)
                    };
                     history.push({
                        role: 'user',
                        parts: [{ functionResponse: { name: 'get_stock_price', response: stockResponse } }]
                    });
                 } else if (call.name === 'generate_image') {
                    const { prompt: imagePrompt } = call.args;
                    const toolCallMsg = addToolCallMessage(`Generating an image of "${imagePrompt as string}"...`);
                    try {
                        const imageResponse = await ai.models.generateImages({
                            model: 'imagen-3.0-generate-002',
                            prompt: imagePrompt as string,
                            config: {
                                numberOfImages: 1,
                                outputMimeType: 'image/png',
                            },
                        });
                
                        if (imageResponse.generatedImages?.length > 0) {
                            const base64Image = imageResponse.generatedImages[0].image.imageBytes;
                            const imageUrl = `data:image/png;base64,${base64Image}`;
                            addModelImageMessage(imageUrl, imagePrompt as string);
                            history.push({
                                role: 'user',
                                parts: [{ functionResponse: { name: 'generate_image', response: { success: true, message: 'Image was created and displayed to the user.' } } }]
                            });
                        } else {
                            throw new Error('Image generation failed to produce an image.');
                        }
                    } catch (imageError) {
                        console.error("Image generation failed:", imageError);
                        const errorMessage = imageError instanceof Error ? imageError.message : 'An unknown error occurred.';
                         history.push({
                            role: 'user',
                            parts: [{ functionResponse: { name: 'generate_image', response: { success: false, error: errorMessage } } }]
                        });
                    }
                    toolCallMsg.remove();
                 }
            } else {
                 isDone = true;
                 history.push({ role: 'model', parts: [{ text: fullText }] });
                 if (wantsRecipe) {
                    try {
                        const recipeJson = JSON.parse(fullText);
                        addRecipeCard(recipeJson);
                        speak(`Here is a recipe for ${recipeJson.recipeName}. ${recipeJson.description || ''}`);
                    } catch (e) {
                        console.error("Failed to parse recipe JSON:", e);
                        modelResponseEl = addMessage('model', `I tried to find a recipe, but the result wasn't in the correct format. Here's the raw data:\n\n\`\`\`json\n${fullText}\n\`\`\``);
                        addCodeCopyButtons(modelResponseEl);
                    }
                 } else if (modelResponseEl) {
                    addCodeCopyButtons(modelResponseEl);
                    if (groundingMetadata?.groundingChunks) {
                        renderGroundingSources(groundingMetadata.groundingChunks, modelResponseEl);
                    }
                 } else if (fullText.trim()) {
                    modelResponseEl = addMessage('model', fullText);
                    addCodeCopyButtons(modelResponseEl);
                 }
                 processSpeech('', true); // Flush any remaining speech
            }

        } catch (error) {
            isDone = true;
            loadingIndicator.remove();
            const message = error instanceof Error ? error.message : 'An unknown error occurred.';
            addMessage('model', `Error: ${message}`);
            console.error(error);
        }
    }
    setBusyState(false);
    promptInput.focus();
  });

  imageUpload.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      const { base64, mimeType } = await fileToGenerativePart(file);
      attachedImage = { base64, mimeType };
      imagePreview.src = `data:${mimeType};base64,${base64}`;
      imagePreviewContainer.style.display = 'block';
    }
  });

  removeImageButton.addEventListener('click', () => clearImagePreview());

  promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = `${promptInput.scrollHeight}px`;
  });
}

main();