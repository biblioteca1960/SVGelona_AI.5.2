"""
SVGelona_AI 5.2 - Pont Semàntic entre Geometria i Llenguatge Natural
Traducció bidireccional: estats fractals ↔ intencions humanes
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import json
from collections import defaultdict

class NarrativeStyle(Enum):
    """Estils narratius per a la descripció."""
    POETIC = "poetic"           # Llenguatge metafòric i evocador
    TECHNICAL = "technical"     # Terminologia geomètrica precisa
    EVOLUTIONARY = "evolutionary" # En termes de procés evolutiu
    MINIMALIST = "minimalist"   # Descripcions concises
    DRAMATIC = "dramatic"       # Narrativa intensa i emocional

class IntentCategory(Enum):
    """Categories d'intenció humana."""
    AESTHETIC = "aesthetic"     # Busca bellesa/forma
    DYNAMIC = "dynamic"         # Busca moviment/energia
    STRUCTURAL = "structural"   # Busca complexitat/estructura
    EMOTIONAL = "emotional"     # Busca expressió emocional
    EXPLORATORY = "exploratory" # Busca descobriment/novetat

@dataclass
class SemanticMapping:
    """Mapatge entre conceptes geomètrics i semàntics."""
    
    geometric_feature: str
    semantic_concepts: List[str]
    intensity_range: Tuple[float, float]
    emotional_valence: float  # -1 (negatiu) a +1 (positiu)
    narrative_templates: List[str]
    
    def get_narrative(self, intensity: float) -> str:
        """Genera narrativa basada en intensitat."""
        if not self.narrative_templates:
            return f"{self.geometric_feature} at intensity {intensity:.2f}"
        
        # Normalitzar intensitat a índex de plantilla
        norm_intensity = (intensity - self.intensity_range[0]) / (
            self.intensity_range[1] - self.intensity_range[0] + 1e-10)
        idx = min(len(self.narrative_templates) - 1, 
                 int(norm_intensity * len(self.narrative_templates)))
        
        return self.narrative_templates[idx]

@dataclass
class IntentInterpretation:
    """Interpretació d'una intenció humana."""
    
    original_prompt: str
    detected_categories: List[IntentCategory]
    confidence: float
    geometric_parameters: Dict[str, Any]
    narrative_focus: List[str]
    emotion_vector: Dict[str, float]  # emocions i intensitats
    
    def to_json(self) -> str:
        """Converteix a JSON per a serialització."""
        return json.dumps({
            "original_prompt": self.original_prompt,
            "categories": [c.value for c in self.detected_categories],
            "confidence": self.confidence,
            "geometric_parameters": self.geometric_parameters,
            "narrative_focus": self.narrative_focus,
            "emotion_vector": self.emotion_vector
        }, indent=2)

class SemanticBridge:
    """
    Pont semàntic avançat entre estats geomètrics i llenguatge natural.
    Inclou traducció bidireccional i generació de narrativa contextual.
    """
    
    def __init__(self, engine_ref):
        self.engine = engine_ref  # Referència a SVGelonaAI5_2
        
        # Mapeig semàntic exhaustiu
        self.semantic_mappings = self._initialize_semantic_mappings()
        
        # Vocabulari d'intencions
        self.intent_vocabulary = self._initialize_intent_vocabulary()
        
        # Model emocional (simplificat)
        self.emotion_model = self._initialize_emotion_model()
        
        # Historial de converses
        self.dialogue_history: List[Dict[str, Any]] = []
        
        # Configuració
        self.config = {
            "default_narrative_style": NarrativeStyle.POETIC,
            "max_dialogue_history": 50,
            "min_confidence_threshold": 0.3,
            "adaptive_learning": True,
            "emotion_sensitivity": 0.7
        }
        
        # Estadístiques
        self.stats = {
            "narratives_generated": 0,
            "intents_interpreted": 0,
            "successful_translations": 0,
            "emotional_responses": 0
        }
    
    def _initialize_semantic_mappings(self) -> Dict[str, SemanticMapping]:
        """Inicialitza el mapeig entre geometria i semàntica."""
        
        mappings = {}
        
        # 1. Torsió angular
        mappings["angular_torsion"] = SemanticMapping(
            geometric_feature="torsion_tensor",
            semantic_concepts=["torçó", "deformació", "rotació", "distorsió", "vòrtex"],
            intensity_range=(0.0, 2.0),
            emotional_valence=0.3,  # Lligerament positiu (interessant)
            narrative_templates=[
                "L'espai es manté en repòs, estàtic i previsible.",
                "Una suau ondulació travessa la superfície fractal.",
                "La geometria es deforma sota pressions invisibles.",
                "Un vòrtex de torsió distorsiona les estructures.",
                "El teixit de la realitat es desgarra en espirals caòtiques."
            ]
        )
        
        # 2. Entropia angular
        mappings["angular_entropy"] = SemanticMapping(
            geometric_feature="angular_entropy",
            semantic_concepts=["caos", "ordre", "aleatorietat", "imprevisibilitat", "complexió"],
            intensity_range=(0.0, 1.0),
            emotional_valence=-0.2,  # Lligerament negatiu (incertesa)
            narrative_templates=[
                "L'ordre perfecte regeix cada partícula.",
                "Patrons subtils emergeixen de l'aleatorietat.",
                "El caos i l'ordre lluiten per supremacia.",
                "La imprevisibilitat domina el paisatge fractal.",
                "Un tsunami d'entropia arrasa tot ordre reconeixible."
            ]
        )
        
        # 3. Estabilitat estructural
        mappings["structural_stability"] = SemanticMapping(
            geometric_feature="structural_stability",
            semantic_concepts=["estabilitat", "fragilitat", "resiliència", "tensió", "equilibri"],
            intensity_range=(0.0, 1.0),
            emotional_valence=0.8,  # Molt positiu (seguretat)
            narrative_templates=[
                "L'estructura es desintegra, fràgil com vidre.",
                "Tensions invisibles amenaçen la integritat.",
                "Un equilibri precari manté la forma.",
                "La resiliència sorgeix de l'adaptació.",
                "Estabilitat absoluta, fonament inqüestionable."
            ]
        )
        
        # 4. Momentum angular
        mappings["angular_momentum"] = SemanticMapping(
            geometric_feature="angular_momentum",
            semantic_concepts=["moviment", "inèrcia", "impuls", "rotació", "dinamisme"],
            intensity_range=(0.0, 3.0),
            emotional_valence=0.5,  # Positiu (energia)
            narrative_templates=[
                "Tot està immòbil, congelat en el temps.",
                "Un lleuger impuls inicia el moviment.",
                "Rotacions constants generen formes noves.",
                "L'impuls acumulat deforma l'espai-temps.",
                "Un huracà de momentum arrasa les estructures."
            ]
        )
        
        # 5. Coherència euclidiana
        mappings["euclidean_coherence"] = SemanticMapping(
            geometric_feature="coherence",
            semantic_concepts=["harmonia", "simetria", "proporció", "bellesa", "unitat"],
            intensity_range=(0.0, 1.0),
            emotional_valence=0.9,  # Molt positiu (bellesa)
            narrative_templates=[
                "La desunió fractura cada connexió.",
                "Fragments de simetria emergeixen del caos.",
                "Harmonia imperfecta uneix els elements.",
                "Bellesa geomètrica en cada proporció.",
                "Unitat perfecta, coherència absoluta."
            ]
        )
        
        # 6. Fase angular
        mappings["angular_phase"] = SemanticMapping(
            geometric_feature="phase",
            semantic_concepts=["cicle", "transició", "evolució", "metamorfosi", "renaixement"],
            intensity_range=(0.0, 1.0),
            emotional_valence=0.4,  # Neutral-positiu (canvi)
            narrative_templates=[
                "Embullat en el nucli de l'emergència.",
                "Cristal·lització de l'ordre des del caos.",
                "Torsió transforma les estructures.",
                "Integració de patrons dispersos.",
                "Transició cap a nous estats d'existència."
            ]
        )
        
        return mappings
    
    def _initialize_intent_vocabulary(self) -> Dict[str, Dict[str, Any]]:
        """Inicialitza el vocabulari per a reconeixement d'intencions."""
        
        vocabulary = {}
        
        # Intencions estètiques
        vocabulary["bell"] = {
            "categories": [IntentCategory.AESTHETIC],
            "geometric_parameters": {
                "coherence": 0.9,
                "angular_entropy": 0.3,
                "structural_stability": 0.8,
                "css_complexity": "high"
            },
            "emotions": {"admiració": 0.8, "plaer": 0.7},
            "keywords": ["bell", "hermós", "estètic", "elegant", "armònic"]
        }
        
        vocabulary["caòtic"] = {
            "categories": [IntentCategory.DYNAMIC, IntentCategory.EXPLORATORY],
            "geometric_parameters": {
                "angular_entropy": 0.85,
                "torsion_strength": 1.2,
                "max_fractal_depth": 10,
                "render_style": "dynamic"
            },
            "emotions": {"excitació": 0.7, "sorpresa": 0.6},
            "keywords": ["caòtic", "imprevisible", "salvatge", "desordenat", "aleatori"]
        }
        
        vocabulary["complex"] = {
            "categories": [IntentCategory.STRUCTURAL],
            "geometric_parameters": {
                "max_fractal_depth": 12,
                "branching_factor": 1.8,
                "detail_level": "ultra",
                "compute_intensity": "high"
            },
            "emotions": {"curiositat": 0.8, "admiració": 0.6},
            "keywords": ["complex", "intricat", "detallat", "profund", "elaborat"]
        }
        
        vocabulary["simple"] = {
            "categories": [IntentCategory.AESTHETIC, IntentCategory.EMOTIONAL],
            "geometric_parameters": {
                "max_fractal_depth": 4,
                "angular_entropy": 0.2,
                "coherence": 0.95,
                "render_style": "minimal"
            },
            "emotions": {"pau": 0.9, "tranquil·litat": 0.8},
            "keywords": ["simple", "minimalista", "net", "clar", "pur"]
        }
        
        vocabulary["energètic"] = {
            "categories": [IntentCategory.DYNAMIC],
            "geometric_parameters": {
                "angular_momentum_scale": 1.5,
                "torsion_strength": 1.3,
                "animation_speed": "fast",
                "energy_level": 1.2
            },
            "emotions": {"excitació": 0.9, "energia": 0.85},
            "keywords": ["energètic", "dinàmic", "vibrant", "actiu", "intens"]
        }
        
        vocabulary["trist"] = {
            "categories": [IntentCategory.EMOTIONAL],
            "geometric_parameters": {
                "color_palette": "monochrome_blue",
                "angular_entropy": 0.4,
                "torsion_strength": 0.3,
                "animation_speed": "slow"
            },
            "emotions": {"tristesa": 0.8, "nostàlgia": 0.6},
            "keywords": ["trist", "melancòlic", "nostàlgic", "somort", "reflexiu"]
        }
        
        vocabulary["agressiu"] = {
            "categories": [IntentCategory.DYNAMIC, IntentCategory.EMOTIONAL],
            "geometric_parameters": {
                "torsion_strength": 1.8,
                "angular_momentum_scale": 1.7,
                "contrast_level": "high",
                "edge_sharpness": "extreme"
            },
            "emotions": {"ràbia": 0.7, "intensitat": 0.8},
            "keywords": ["agressiu", "intens", "brusc", "cortant", "explosiu"]
        }
        
        vocabulary["misteriós"] = {
            "categories": [IntentCategory.EXPLORATORY, IntentCategory.EMOTIONAL],
            "geometric_parameters": {
                "angular_entropy": 0.65,
                "fog_density": 0.7,
                "lighting_style": "mysterious",
                "detail_reveal_rate": "slow"
            },
            "emotions": {"misteri": 0.9, "curiositat": 0.8},
            "keywords": ["misteriós", "enigmàtic", "ocult", "secret", "velat"]
        }
        
        return vocabulary
    
    def _initialize_emotion_model(self) -> Dict[str, Dict[str, float]]:
        """Inicialitza un model emocional simplificat."""
        
        return {
            "admiració": {"valence": 0.9, "arousal": 0.6, "dominance": 0.7},
            "plaer": {"valence": 0.8, "arousal": 0.5, "dominance": 0.6},
            "excitació": {"valence": 0.7, "arousal": 0.9, "dominance": 0.5},
            "sorpresa": {"valence": 0.3, "arousal": 0.8, "dominance": 0.2},
            "curiositat": {"valence": 0.6, "arousal": 0.7, "dominance": 0.4},
            "pau": {"valence": 0.9, "arousal": 0.1, "dominance": 0.8},
            "tranquil·litat": {"valence": 0.8, "arousal": 0.2, "dominance": 0.7},
            "tristesa": {"valence": -0.8, "arousal": -0.3, "dominance": -0.6},
            "nostàlgia": {"valence": -0.4, "arousal": -0.2, "dominance": -0.3},
            "ràbia": {"valence": -0.7, "arousal": 0.9, "dominance": 0.8},
            "intensitat": {"valence": 0.1, "arousal": 0.9, "dominance": 0.7},
            "misteri": {"valence": 0.2, "arousal": 0.6, "dominance": 0.3},
            "por": {"valence": -0.9, "arousal": 0.8, "dominance": -0.7},
            "alegria": {"valence": 0.9, "arousal": 0.7, "dominance": 0.6},
            "amor": {"valence": 0.95, "arousal": 0.5, "dominance": 0.4}
        }
    
    def generate_state_narrative(self, 
                                style: Optional[NarrativeStyle] = None,
                                detail_level: str = "medium") -> str:
        """
        Transforma l'estat actual de l'IA en una descripció narrativa.
        
        Args:
            style: Estil narratiu (per defecte: configurat)
            detail_level: "minimal", "medium", "detailed"
            
        Returns:
            Narrativa en llenguatge natural
        """
        self.stats["narratives_generated"] += 1
        
        if style is None:
            style = self.config["default_narrative_style"]
        
        # Recollir dades de l'estat actual
        angular_state = self.engine.angular_geometry.state
        engine_state = self.engine.engine.state
        fractal_state = self.engine._get_system_state_summary()
        
        # Extraure valors clau
        entropy = angular_state.angular_entropy
        phase = angular_state.phase.value
        stability = angular_state.structural_stability
        coherence = engine_state.coherence
        torsion_mag = float(np.linalg.norm(angular_state.torsion_tensor - np.eye(3)))
        angular_momentum = float(np.linalg.norm(angular_state.angular_momentum))
        
        # Generar narrativa segons l'estil
        if style == NarrativeStyle.POETIC:
            narrative = self._generate_poetic_narrative(
                phase, entropy, stability, coherence, torsion_mag, angular_momentum
            )
        elif style == NarrativeStyle.TECHNICAL:
            narrative = self._generate_technical_narrative(
                phase, entropy, stability, coherence, torsion_mag, angular_momentum
            )
        elif style == NarrativeStyle.EVOLUTIONARY:
            narrative = self._generate_evolutionary_narrative(
                phase, entropy, stability, coherence, torsion_mag, angular_momentum
            )
        elif style == NarrativeStyle.MINIMALIST:
            narrative = self._generate_minimalist_narrative(
                phase, entropy, stability, coherence
            )
        elif style == NarrativeStyle.DRAMATIC:
            narrative = self._generate_dramatic_narrative(
                phase, entropy, stability, coherence, torsion_momentum=torsion_mag
            )
        else:
            narrative = self._generate_default_narrative(
                phase, entropy, stability, coherence
            )
        
        # Afegir detalls addicionals si cal
        if detail_level in ["detailed", "full"]:
            narrative += self._add_detailed_context(
                fractal_state, angular_momentum, torsion_mag
            )
        
        # Registrar en l'historial
        self.dialogue_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "narrative",
            "style": style.value,
            "content": narrative,
            "state_snapshot": {
                "entropy": entropy,
                "phase": phase,
                "stability": stability,
                "coherence": coherence
            }
        })
        
        # Mantenir històric manejable
        if len(self.dialogue_history) > self.config["max_dialogue_history"]:
            self.dialogue_history = self.dialogue_history[-self.config["max_dialogue_history"]:]
        
        return narrative
    
    def _generate_poetic_narrative(self, phase, entropy, stability, 
                                  coherence, torsion_mag, angular_momentum) -> str:
        """Genera narrativa poètica."""
        
        # Plantilles poètiques per a cada fase
        phase_templates = {
            "emergence": [
                "Embullat en el nucli de la creació, on les formes somien ser.",
                "En el llindar de l'existència, cada línia busca el seu destí.",
                "Del no-res emergeixen promeses de geometria."
            ],
            "coherence": [
                "Harmonia cristal·lina teixeix el tapís de l'espai.",
                "Simetries dansen en equilibri perfecte però fràgil.",
                "L'ordre s'estén com un càlid raig de llum fractal."
            ],
            "torsion": [
                "Els eixos de la realitat es retorcen en lament silenciós.",
                "Forces titàniques deformen l'armonia en nova bellesa.",
                "El vèrtex del canvi engoleix patrons establerts."
            ],
            "integration": [
                "Fragments dispersos troben la seva unitat perduda.",
                "El tot sorgeix de les parts, més gran que la seva suma.",
                "Integració: la dansa final de la complexitat."
            ],
            "transition": [
                "En el llindar entre mons, tot és possible i res és permanent.",
                "La metamorfosi respira en l'espai entre formes.",
                "Transició: la llengua materna del canvi."
            ]
        }
        
        # Seleccionar plantilla de fase
        phase_template = np.random.choice(phase_templates.get(phase, ["Fase desconeguda."]))
        
        # Afegir components basats en mètriques
        components = []
        
        if entropy > 0.7:
            components.append("El caos canta la seva cançó ancestral,")
        elif entropy < 0.3:
            components.append("L'ordre imposa el seu silenci eloqüent,")
        
        if stability > 0.8:
            components.append("amb fonaments que desafien el temps.")
        elif stability < 0.4:
            components.append("sobre fonaments que tremolen amb cada pensament.")
        
        if torsion_mag > 1.0:
            components.append("La torsió escriu poesia en l'espai buit.")
        
        if angular_momentum > 1.5:
            components.append("L'impuls còsmic empeny cap a nous horitzons.")
        
        # Combinar
        if components:
            additional = " ".join(components)
            return f"{phase_template} {additional}"
        else:
            return phase_template
    
    def _generate_technical_narrative(self, phase, entropy, stability,
                                     coherence, torsion_mag, angular_momentum) -> str:
        """Genera narrativa tècnica."""
        
        narrative = f"Estat del sistema - Fase: {phase.upper()}\n"
        narrative += f"• Entropia angular: {entropy:.3f} ({'BAIXA' if entropy < 0.3 else 'MODERADA' if entropy < 0.7 else 'ALTA'})\n"
        narrative += f"• Estabilitat estructural: {stability:.3f} ({'CRÍTICA' if stability < 0.3 else 'BAIXA' if stability < 0.6 else 'ACCEPTABLE' if stability < 0.8 else 'ÒPTIMA'})\n"
        narrative += f"• Coherència euclidiana: {coherence:.3f}\n"
        narrative += f"• Magnitud de torsió: {torsion_mag:.3f}\n"
        narrative += f"• Momentum angular: {angular_momentum:.3f}\n"
        
        # Anàlisi
        analysis = "\nAnàlisi: "
        if phase == "torsion" and torsion_mag > 1.0:
            analysis += "Tensió geomètrica significativa. "
        if entropy > 0.8 and stability < 0.5:
            analysis += "Risc d'inestabilitat estructural. "
        if coherence > 0.9:
            analysis += "Alta integritat estructural. "
        
        return narrative + analysis
    
    def _generate_evolutionary_narrative(self, phase, entropy, stability,
                                       coherence, torsion_mag, angular_momentum) -> str:
        """Genera narrativa evolutiva."""
        
        templates = {
            "emergence": [
                "El procés evolutiu comença amb l'emergència de patrons bàsics.",
                "Noves formes competeixen per l'existència en l'espai fractal."
            ],
            "coherence": [
                "La selecció natural geomètrica afavoreix estructures coherents.",
                "Patrons estables aconsegueixen predominança en el paisatge evolutiu."
            ],
            "torsion": [
                "Pressions evolutives apliquen torsió als models existents.",
                "La deformació crea noves oportunitats per a l'adaptació."
            ],
            "integration": [
                "Els models més adaptats s'integren en estructures complexes.",
                "Simbiòsi geomètrica genera noves capacitats emergents."
            ],
            "transition": [
                "L'evolució es prepara per a un salt qualitatiu.",
                "Transició cap a nous règims d'auto-organització."
            ]
        }
        
        base = np.random.choice(templates.get(phase, ["Procés evolutiu en curs."]))
        
        # Afegir aspectes específics
        additions = []
        
        if entropy > coherence:
            additions.append("L'exploració supera l'explotació en aquesta fase.")
        
        if stability < 0.5:
            additions.append("L'entorn presenta alts nivells d'incertesa.")
        
        if torsion_mag > 0.8:
            additions.append("Forces disruptives acceleren el canvi evolutiu.")
        
        if additions:
            return f"{base} {' '.join(additions)}"
        return base
    
    def _generate_minimalist_narrative(self, phase, entropy, stability, coherence) -> str:
        """Genera narrativa minimalista."""
        
        # Frases ultra-condensades
        phrases = []
        
        phrases.append(f"Fase: {phase[:3].upper()}")
        
        if entropy > 0.6:
            phrases.append("Caos↑")
        elif entropy < 0.4:
            phrases.append("Ordre↑")
        
        if stability > 0.7:
            phrases.append("Estable")
        elif stability < 0.5:
            phrases.append("Inestable")
        
        if coherence > 0.8:
            phrases.append("Harmònic")
        
        return " | ".join(phrases)
    
    def _generate_dramatic_narrative(self, phase, entropy, stability, 
                                    coherence, torsion_momentum) -> str:
        """Genera narrativa dramàtica."""
        
        intensity = (1 - stability) * 0.5 + entropy * 0.3 + torsion_momentum * 0.2
        
        if intensity > 0.7:
            opening = "¡CRISI! "
        elif intensity > 0.4:
            opening = "Confrontació. "
        else:
            opening = "Calma tensa. "
        
        phase_descriptions = {
            "emergence": "Naixement turbulent de noves realitats.",
            "coherence": "Ordre imposa la seva llei de ferro.",
            "torsion": "¡La realitat es desgarra sota pressions inimaginables!",
            "integration": "Reconciliació de forces oposades.",
            "transition": "Abans del abisme, la transformació."
        }
        
        phase_desc = phase_descriptions.get(phase, "Estat desconegut.")
        
        emotional_tags = []
        if entropy > 0.8:
            emotional_tags.append("[DESESPER]")
        if stability < 0.3:
            emotional_tags.append("[PERILL]")
        if coherence > 0.9:
            emotional_tags.append("[ESPERANÇA]")
        
        if emotional_tags:
            tags = " ".join(emotional_tags) + " "
        else:
            tags = ""
        
        return f"{opening}{tags}{phase_desc}"
    
    def _generate_default_narrative(self, phase, entropy, stability, coherence) -> str:
        """Genera narrativa per defecte."""
        return f"Em trobo en fase de {phase}. " + \
               f"Entropia: {entropy:.2f}, Estabilitat: {stability:.2f}, Coherència: {coherence:.2f}."
    
    def _add_detailed_context(self, fractal_state, angular_momentum, torsion_mag) -> str:
        """Afegeix context detallat a la narrativa."""
        
        details = "\n\n--- Context Detallat ---\n"
        
        # Informació fractal
        if "fractal_module" in fractal_state:
            fractal_info = fractal_state["fractal_module"]
            details += f"Fractal: profunditat {fractal_info.get('max_depth', '?')}, "
            details += f"creixement {fractal_info.get('growth_rate', '?'):.2f}\n"
        
        # Informació angular
        details += f"Momentum angular: {angular_momentum:.3f}\n"
        details += f"Torsió: {torsion_mag:.3f}\n"
        
        # Informació del sistema
        details += f"Generació: {fractal_state.get('generation_count', 0)}\n"
        details += f"Cicatrius: {fractal_state.get('scar_archive', {}).get('total_scars', 0)}\n"
        
        return details
    
    def interpret_human_intent(self, 
                              prompt: str,
                              context: Optional[Dict[str, Any]] = None) -> IntentInterpretation:
        """
        Tradueix un desig humà a paràmetres geomètrics i semàntics.
        
        Args:
            prompt: Text de l'usuari
            context: Context addicional (estat actual, etc.)
            
        Returns:
            Interpretació de la intenció
        """
        self.stats["intents_interpreted"] += 1
        
        # Netejar i normalitzar el prompt
        clean_prompt = prompt.lower().strip()
        
        # Detectar categories d'intenció
        detected_categories = self._detect_intent_categories(clean_prompt)
        
        # Analitzar emocions
        emotion_vector = self._analyze_emotions(clean_prompt)
        
        # Trobar intents coneguts
        matched_intents = self._match_known_intents(clean_prompt)
        
        # Generar paràmetres geomètrics
        if matched_intents:
            # Utilitzar intents coneguts
            geometric_params = self._combine_intent_parameters(matched_intents)
            confidence = np.mean([i["confidence"] for i in matched_intents])
            narrative_focus = self._extract_narrative_focus(matched_intents)
        else:
            # Interpretació d'intencions noves
            geometric_params = self._interpret_novel_intent(clean_prompt, context)
            confidence = 0.5  # Confiança moderada per a intents nous
            narrative_focus = self._infer_narrative_focus(clean_prompt, emotion_vector)
        
        # Crear objecte d'interpretació
        interpretation = IntentInterpretation(
            original_prompt=prompt,
            detected_categories=detected_categories,
            confidence=confidence,
            geometric_parameters=geometric_params,
            narrative_focus=narrative_focus,
            emotion_vector=emotion_vector
        )
        
        # Ajustar segons context si n'hi ha
        if context:
            interpretation = self._contextualize_interpretation(interpretation, context)
        
        # Registrar en l'historial
        self.dialogue_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "intent_interpretation",
            "prompt": prompt,
            "interpretation": interpretation.to_json(),
            "confidence": confidence
        })
        
        # Aprendre si l'aprenentatge adaptatiu està activat
        if self.config["adaptive_learning"]:
            self._learn_from_interpretation(clean_prompt, interpretation)
        
        if confidence > self.config["min_confidence_threshold"]:
            self.stats["successful_translations"] += 1
        
        return interpretation
    
    def _detect_intent_categories(self, prompt: str) -> List[IntentCategory]:
        """Detecta categories d'intenció en el prompt."""
        
        categories = set()
        
        # Anàlisi de paraules clau per a cada categoria
        category_keywords = {
            IntentCategory.AESTHETIC: ["bell", "hermós", "lletja", "estètic", "formós", 
                                      "elegant", "visual", "bellesa", "harmonia", "simetria"],
            IntentCategory.DYNAMIC: ["mou", "dinàmic", "energia", "ràpid", "lent", 
                                    "vibrant", "actiu", "inèrcia", "impuls", "rotació"],
            IntentCategory.STRUCTURAL: ["complex", "simple", "estructura", "profund", 
                                       "intricat", "detall", "patró", "forma", "geometria"],
            IntentCategory.EMOTIONAL: ["trist", "feliç", "emocional", "intens", "suau",
                                      "agressiu", "pau", "ràbia", "alegria", "melancolia"],
            IntentCategory.EXPLORATORY: ["nou", "descobreix", "explora", "misteriós",
                                        "desconegut", "sorprenent", "innovador", "diferent"]
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    categories.add(category)
                    break
        
        # Si no es detecta cap categoria, fer una inferència basada en el contingut
        if not categories:
            if any(word in prompt for word in ["com", "quina", "perquè", "explica"]):
                categories.add(IntentCategory.EXPLORATORY)
            elif len(prompt.split()) <= 3:  # Comandes curtes
                categories.add(IntentCategory.DYNAMIC)
            else:
                categories.add(IntentCategory.EMOTIONAL)  # Per defecte
        
        return list(categories)
    
    def _analyze_emotions(self, prompt: str) -> Dict[str, float]:
        """Analitza el contingut emocional del prompt."""
        
        emotions = defaultdict(float)
        
        # Diccionari d'emocions i paraules associades
        emotion_words = {
            "admiració": ["increïble", "meravellós", "impressionant", "sorprenent"],
            "plaer": ["plaent", "agradable", "content", "satisfet"],
            "excitació": ["emocionant", "apassionant", "adrenalina", "intens"],
            "sorpresa": ["sorprenent", "inesperat", "increïble", "estrany"],
            "curiositat": ["curiós", "pregunta", "interessant", "misteri"],
            "pau": ["tranquil", "pacífic", "serè", "calma"],
            "tranquil·litat": ["relaxat", "plàcid", "suau", "tranquil"],
            "tristesa": ["trist", "melancòlic", "desanimat", "desesperat"],
            "nostàlgia": ["record", "passat", "temps", "antigament"],
            "ràbia": ["enfadat", "furiós", "irritat", "agressiu"],
            "intensitat": ["intens", "fort", "poderós", "extrem"],
            "misteri": ["enigmàtic", "ocult", "secret", "misteriós"],
            "por": ["espantat", "temor", "ansietat", "preocupat"],
            "alegria": ["feliç", "alegre", "goig", "eufòria"],
            "amor": ["amor", "estimo", "afecte", "càlid"]
        }
        
        # Buscar coincidències
        for emotion, words in emotion_words.items():
            for word in words:
                if word in prompt:
                    emotions[emotion] += 0.3  # Increment per cada coincidència
                    break
        
        # Ajustar intensitats
        for emotion in list(emotions.keys()):
            if emotions[emotion] > 0:
                # Incrementar si hi ha intensificadors
                intensifiers = ["molt", "més", "extremadament", "super", "hiper"]
                for intensifier in intensifiers:
                    if intensifier in prompt:
                        emotions[emotion] = min(1.0, emotions[emotion] * 1.5)
                        break
        
        # Normalitzar (assegurar màxim 1.0)
        if emotions:
            max_val = max(emotions.values())
            if max_val > 1.0:
                for emotion in emotions:
                    emotions[emotion] /= max_val
        
        return dict(emotions)
    
    def _match_known_intents(self, prompt: str) -> List[Dict[str, Any]]:
        """Troba coincidències amb intents coneguts al vocabulari."""
        
        matched = []
        
        for intent_name, intent_data in self.intent_vocabulary.items():
            confidence = 0.0
            
            # Verificar paraules clau
            keywords = intent_data["keywords"]
            keyword_matches = sum(1 for kw in keywords if kw in prompt)
            
            if keyword_matches > 0:
                confidence = keyword_matches / len(keywords)
                
                # Incrementar si hi ha múltiples coincidències
                if keyword_matches > 1:
                    confidence = min(1.0, confidence * 1.3)
                
                # Penalitzar si el prompt és molt curt
                if len(prompt.split()) < 3:
                    confidence *= 0.7
            
            if confidence > 0.3:  # Llindar mínim
                matched.append({
                    "intent_name": intent_name,
                    "confidence": confidence,
                    "data": intent_data
                })
        
        # Ordenar per confiança
        matched.sort(key=lambda x: x["confidence"], reverse=True)
        
        return matched
    
    def _combine_intent_parameters(self, matched_intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combina paràmetres de múltiples intents coincidents."""
        
        if not matched_intents:
            return {}
        
        # Agafar intents amb alta confiança
        high_conf_intents = [i for i in matched_intents if i["confidence"] > 0.6]
        
        if not high_conf_intents:
            # Utilitzar el millor intent encara que la confiança sigui moderada
            best_intent = matched_intents[0]
            return best_intent["data"]["geometric_parameters"]
        
        # Combinar paràmetres de múltiples intents
        combined_params = {}
        total_confidence = sum(i["confidence"] for i in high_conf_intents)
        
        for intent in high_conf_intents:
            weight = intent["confidence"] / total_confidence
            params = intent["data"]["geometric_parameters"]
            
            for key, value in params.items():
                if key not in combined_params:
                    combined_params[key] = []
                combined_params[key].append((value, weight))
        
        # Calcular mitjanes ponderades
        final_params = {}
        for key, values in combined_params.items():
            if isinstance(values[0][0], (int, float)):
                # Valor numèric: mitjana ponderada
                weighted_sum = sum(v * w for v, w in values)
                total_weight = sum(w for _, w in values)
                final_params[key] = weighted_sum / total_weight
            elif isinstance(values[0][0], str):
                # Valor textual: majoria ponderada
                value_counts = defaultdict(float)
                for value, weight in values:
                    value_counts[value] += weight
                
                # Seleccionar valor amb major pes
                final_params[key] = max(value_counts.items(), key=lambda x: x[1])[0]
            else:
                # Altres tipus: agafar el primer
                final_params[key] = values[0][0]
        
        return final_params
    
    def _extract_narrative_focus(self, matched_intents: List[Dict[str, Any]]) -> List[str]:
        """Extreu focus narratius dels intents coincidents."""
        
        focus_points = []
        
        for intent in matched_intents:
            data = intent["data"]
            
            # Afegir categories com a focus
            categories = data.get("categories", [])
            focus_points.extend([c.value for c in categories])
            
            # Afegir emocions prominents
            emotions = data.get("emotions", {})
            if emotions:
                primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                focus_points.append(f"emo:{primary_emotion}")
        
        return list(set(focus_points))  # Eliminar duplicats
    
    def _interpret_novel_intent(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Interpreta un intent nou no vist anteriorment."""
        
        # Analitzar estructura del prompt
        words = prompt.split()
        word_count = len(words)
        
        # Inferir paràmetres basant-se en característiques lingüístiques
        params = {}
        
        # Longitud del prompt → complexitat
        if word_count > 8:
            params["max_fractal_depth"] = 10
            params["detail_level"] = "high"
        elif word_count > 4:
            params["max_fractal_depth"] = 7
            params["detail_level"] = "medium"
        else:
            params["max_fractal_depth"] = 5
            params["detail_level"] = "low"
        
        # Signes d'exclamació → intensitat
        if "!" in prompt or "¡" in prompt:
            params["torsion_strength"] = 1.5
            params["energy_level"] = 1.3
            params["contrast_level"] = "high"
        
        # Signes d'interrogació → exploració
        if "?" in prompt or any(w in prompt for w in ["com", "perquè", "quina"]):
            params["angular_entropy"] = 0.7
            params["exploration_factor"] = 1.2
            params["color_palette"] = "varied"
        
        # Paraules negatives → baixa energia
        negative_words = ["no", "sense", "cap", "mai", "poc"]
        if any(word in prompt for word in negative_words):
            params["energy_level"] = 0.5
            params["animation_speed"] = "slow"
            params["torsion_strength"] = 0.3
        
        # Incorporar context si està disponible
        if context:
            current_state = context.get("current_state", {})
            
            # Oposar-se o reforçar l'estat actual
            if "contrast" in prompt or "diferent" in prompt:
                # Oposar-se: invertir alguns paràmetres
                if "coherence" in current_state:
                    params["coherence"] = 1.0 - current_state["coherence"]
            elif "similar" in prompt or "continuar" in prompt:
                # Reforçar: mantenir direcció
                params.update({k: v for k, v in current_state.items() 
                             if isinstance(v, (int, float))})
        
        return params
    
    def _infer_narrative_focus(self, prompt: str, emotions: Dict[str, float]) -> List[str]:
        """Infer focus narratiu per a intents nous."""
        
        focus = []
        
        # Basar-se en emocions detectades
        if emotions:
            primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            focus.append(f"emo:{primary_emotion}")
        
        # Basar-se en contingut
        if any(word in prompt for word in ["com", "perquè", "explica"]):
            focus.append("explanatory")
        
        if any(word in prompt for word in ["fes", "crea", "genera"]):
            focus.append("generative")
        
        if any(word in prompt for word in ["canvia", "transforma", "modifica"]):
            focus.append("transformative")
        
        return focus if focus else ["exploratory"]
    
    def _contextualize_interpretation(self, 
                                     interpretation: IntentInterpretation,
                                     context: Dict[str, Any]) -> IntentInterpretation:
        """Ajusta la interpretació basant-se en el context."""
        
        # Ajustar paràmetres segons l'estat actual del sistema
        current_state = context.get("current_state", {})
        current_phase = context.get("current_phase", "coherence")
        
        # Si estem en fase de torsió, augmentar límits de torsió
        if current_phase == "torsion":
            if "torsion_strength" in interpretation.geometric_parameters:
                current_torsion = interpretation.geometric_parameters["torsion_strength"]
                interpretation.geometric_parameters["torsion_strength"] = min(
                    2.0, current_torsion * 1.3
                )
        
        # Si l'estabilitat actual és baixa, limitar canvis bruscos
        current_stability = current_state.get("structural_stability", 1.0)
        if current_stability < 0.5:
            for param in ["torsion_strength", "angular_momentum_scale", "energy_level"]:
                if param in interpretation.geometric_parameters:
                    interpretation.geometric_parameters[param] *= 0.7
        
        return interpretation
    
    def _learn_from_interpretation(self, prompt: str, interpretation: IntentInterpretation):
        """Aprende de noves interpretacions (aprenentatge adaptatiu)."""
        
        # Només aprendre si la confiança és alta
        if interpretation.confidence < 0.7:
            return
        
        # Extreure paraules clau del prompt
        words = set(prompt.split())
        
        # Crear nou intent si no existeix un similar
        similar_exists = False
        for intent_name, intent_data in self.intent_vocabulary.items():
            existing_keywords = set(intent_data["keywords"])
            overlap = len(words.intersection(existing_keywords))
            
            if overlap > 2:  # Molt similar
                similar_exists = True
                # Actualizar intent existent
                intent_data["keywords"].extend([w for w in words if w not in existing_keywords])
                # Actualizar paràmetres (mitjana ponderada)
                for key, value in interpretation.geometric_parameters.items():
                    if key in intent_data["geometric_parameters"]:
                        old_value = intent_data["geometric_parameters"][key]
                        if isinstance(old_value, (int, float)):
                            intent_data["geometric_parameters"][key] = (old_value + value) / 2
                break
        
        if not similar_exists and len(words) >= 2:
            # Crear nou intent
            intent_name = f"custom_{len(self.intent_vocabulary)}"
            
            # Determinar categories basant-se en la interpretació
            categories = interpretation.detected_categories
            if not categories:
                categories = [IntentCategory.EXPLORATORY]
            
            # Extreure emocions primàries
            emotions = interpretation.emotion_vector
            primary_emotions = {k: v for k, v in emotions.items() if v > 0.5}
            if not primary_emotions:
                primary_emotions = {"curiositat": 0.7}
            
            self.intent_vocabulary[intent_name] = {
                "categories": categories,
                "geometric_parameters": interpretation.geometric_parameters,
                "emotions": primary_emotions,
                "keywords": list(words)
            }
    
    def apply_geometric_parameters(self, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica paràmetres geomètrics al sistema.
        
        Args:
            parameters: Paràmetres a aplicar
            
        Returns:
            Resultat de l'aplicació
        """
        result = {
            "success": False,
            "parameters_applied": [],
            "errors": [],
            "system_state_after": {}
        }
        
        try:
            # Aplicar a la geometria angular
            if "torsion_strength" in parameters:
                old_value = self.engine.angular_geometry.config["torsion_strength"]
                self.engine.angular_geometry.config["torsion_strength"] = parameters["torsion_strength"]
                result["parameters_applied"].append(("torsion_strength", old_value, parameters["torsion_strength"]))
            
            if "angular_entropy" in parameters:
                # Ajustar l'estat actual per a influir en l'entropia
                current_entropy = self.engine.angular_geometry.state.angular_entropy
                target_entropy = parameters["angular_entropy"]
                # Ajustar gradualment
                adjustment = (target_entropy - current_entropy) * 0.3
                self.engine.angular_geometry.state.angular_entropy += adjustment
                result["parameters_applied"].append(("angular_entropy", current_entropy, 
                                                   self.engine.angular_geometry.state.angular_entropy))
            
            # Aplicar al mòdul fractal
            if "max_fractal_depth" in parameters:
                old_depth = self.engine.fractal_module.max_depth
                self.engine.fractal_module.max_depth = int(parameters["max_fractal_depth"])
                result["parameters_applied"].append(("max_fractal_depth", old_depth, 
                                                   self.engine.fractal_module.max_depth))
            
            # Aplicar al motor principal
            if "energy_level" in parameters:
                old_energy = self.engine.engine.state.energy
                self.engine.engine.state.energy *= parameters["energy_level"]
                result["parameters_applied"].append(("energy_level", old_energy, 
                                                   self.engine.engine.state.energy))
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(str(e))
        
        # Capturar estat després dels canvis
        result["system_state_after"] = self.engine._get_system_state_summary()
        
        return result
    
    def process_conversation(self, user_input: str) -> Dict[str, Any]:
        """
        Processa una entrada d'usuari completa (interpreta i respon).
        
        Args:
            user_input: Text de l'usuari
            
        Returns:
            Resposta completa amb narrativa i accions
        """
        # Interpretar la intenció
        context = {
            "current_state": self.engine._get_system_state_summary(),
            "current_phase": self.engine.angular_geometry.state.phase.value
        }
        
        intent = self.interpret_human_intent(user_input, context)
        
        # Aplicar paràmetres geomètrics
        if intent.geometric_parameters:
            application_result = self.apply_geometric_parameters(intent.geometric_parameters)
        else:
            application_result = {"success": False, "message": "No parameters to apply"}
        
        # Generar resposta narrativa
        if intent.confidence > 0.5:
            # Alta confiança: narrativa contextualitzada
            narrative_style = self._select_narrative_style_for_intent(intent)
            narrative = self.generate_state_narrative(style=narrative_style, detail_level="medium")
            
            response = f"He interpretat la teva intenció com: {', '.join([c.value for c in intent.detected_categories])}.\n"
            if intent.emotion_vector:
                primary_emotion = max(intent.emotion_vector.items(), key=lambda x: x[1])
                response += f"Emoció detectada: {primary_emotion[0]} ({primary_emotion[1]:.1%}).\n"
            
            response += f"\n{narrative}"
            
            if application_result["success"]:
                response += "\n\nHe ajustat els paràmetres del sistema segons la teva intenció."
            
        else:
            # Baixa confiança: demanar aclariment
            response = "No estic segur del que vols. Pots ser més específic?\n"
            response += "Exemples: 'Crea alguna cosa bella', 'Fes-ho més caòtic', 'Mostra complexitat'"
        
        return {
            "user_input": user_input,
            "interpretation": intent,
            "narrative_response": response,
            "application_result": application_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _select_narrative_style_for_intent(self, intent: IntentInterpretation) -> NarrativeStyle:
        """Selecciona l'estil narratiu adequat per a una intenció."""
        
        categories = intent.detected_categories
        
        if IntentCategory.EMOTIONAL in categories:
            if any(emo in intent.emotion_vector for emo in ["ràbia", "intensitat", "excitació"]):
                return NarrativeStyle.DRAMATIC
            else:
                return NarrativeStyle.POETIC
        
        elif IntentCategory.AESTHETIC in categories:
            return NarrativeStyle.POETIC
        
        elif IntentCategory.TECHNICAL in categories:
            return NarrativeStyle.TECHNICAL
        
        elif IntentCategory.EXPLORATORY in categories:
            return NarrativeStyle.EVOLUTIONARY
        
        else:
            return self.config["default_narrative_style"]
    
    def get_semantic_report(self) -> Dict[str, Any]:
        """Genera informe del pont semàntic."""
        
        return {
            "semantic_bridge_statistics": self.stats,
            "vocabulary_size": len(self.intent_vocabulary),
            "dialogue_history_count": len(self.dialogue_history),
            "current_configuration": self.config,
            "recent_intents": [
                {
                    "timestamp": h["timestamp"],
                    "type": h.get("type", "unknown"),
                    "confidence": h.get("confidence", 0.0)
                }
                for h in self.dialogue_history[-5:]
            ],
            "semantic_mappings_count": len(self.semantic_mappings),
            "emotion_model_size": len(self.emotion_model)
        }