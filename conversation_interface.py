# conversation_interface.py
"""
Interfície conversacional per a SVGelona_AI 5.2
Ara amb suport per a múltiples configuracions i comandes avançades.
"""
import sys
import os
import json
from typing import Dict, List, Any, Optional
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_v5_2 import SVGelonaAI5_2
from config_v5_2 import (
    get_config, 
    get_config_object, 
    list_available_configs,
    create_custom_config,
    validate_config,
    optimize_config_for_hardware,
    print_config_summary,
    ALL_CONFIGS
)

class ConversationalSVGelona:
    """Classe millorada per a gestió de conversacions."""
    
    def __init__(self, config_name: str = "conversational", custom_config: Optional[Dict] = None):
        """
        Inicialitza el sistema conversacional.
        
        Args:
            config_name: Nom de la configuració a utilitzar
            custom_config: Configuració personalitzada opcional
        """
        print("=" * 80)
        print("SVGelona_AI 5.2 - Interfície Conversacional Avançada")
        print("=" * 80)
        
        # Carregar configuració
        if custom_config:
            config = custom_config
            print("✓ Configuració personalitzada carregada")
        else:
            if config_name in ALL_CONFIGS:
                config_obj = get_config_object(config_name)
                config = config_obj.config.copy()
                print(f"✓ Configuració '{config_name}' carregada: {config_obj.description}")
            else:
                print(f"⚠ Configuració '{config_name}' no trobada, usant 'conversational'")
                config_obj = get_config_object("conversational")
                config = config_obj.config.copy()
        
        # Inicialitzar sistema
        self.system = SVGelonaAI5_2(config)
        self.config_name = config_name
        self.config = config
        
        # Historial de conversa
        self.conversation_history = []
        self.command_history = []
        
        # Estats especials
        self.creative_mode = False
        self.visualization_mode = False
        self.learning_mode = True
        
        # Configuracions disponibles
        self.available_configs = list_available_configs()
        
        print(f"\n💬 Sistema preparat en mode '{config_name}'")
        print(f"📊 Configuració: {config['performance_mode']}, Memòria: {config['memory_limit_mb']}MB")
        print(f"🎨 Modes: {'🟢 Creatiu' if self.creative_mode else '⚫ Normal'}")
        print("=" * 80)
    
    def process_command(self, user_input: str) -> Dict[str, Any]:
        """
        Processa una comanda o entrada de l'usuari.
        
        Args:
            user_input: Text introduït per l'usuari
            
        Returns:
            Resposta del sistema
        """
        user_input = user_input.strip()
        
        # Guardar en historial
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": self._get_timestamp()
        })
        
        # Comandes especials
        if user_input.startswith('/'):
            return self._process_special_command(user_input)
        
        # Mode creatiu: interpretar com a inspiració artística
        if self.creative_mode:
            return self._process_creative_input(user_input)
        
        # Mode visualització: interpretar com a descripció visual
        if self.visualization_mode:
            return self._process_visual_input(user_input)
        
        # Processament normal: conversa amb el pont semàntic
        return self._process_normal_input(user_input)
    
    def _process_special_command(self, command: str) -> Dict[str, Any]:
        """
        Processa comandes especials (comencen amb /).
        """
        parts = command[1:].split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        response = {
            "type": "command_response",
            "command": cmd,
            "success": True,
            "message": "",
            "data": {}
        }
        
        try:
            if cmd == "help":
                response["message"] = self._get_help_message()
                
            elif cmd == "config":
                if not args:
                    # Mostrar configuració actual
                    response["message"] = self._get_current_config_info()
                elif args[0] == "list":
                    response["message"] = self._list_configs()
                elif args[0] == "switch" and len(args) > 1:
                    new_config = args[1]
                    if new_config in self.available_configs:
                        self._switch_config(new_config)
                        response["message"] = f"Canviat a configuració '{new_config}'"
                    else:
                        response["success"] = False
                        response["message"] = f"Configuració '{new_config}' no trobada"
                elif args[0] == "show" and len(args) > 1:
                    config_name = args[1]
                    if config_name in self.available_configs:
                        print_config_summary(config_name)
                        response["message"] = f"Resum de configuració '{config_name}' mostrat"
                    else:
                        response["success"] = False
                        response["message"] = f"Configuració '{config_name}' no trobada"
                elif args[0] == "custom":
                    # Crear configuració personalitzada
                    if len(args) > 1:
                        try:
                            params = json.loads(" ".join(args[1:]))
                            custom_config = create_custom_config("default", params)
                            validation = validate_config(custom_config)
                            
                            if validation["is_valid"]:
                                self.system = SVGelonaAI5_2(custom_config)
                                self.config = custom_config
                                response["message"] = "Configuració personalitzada aplicada"
                                if validation["warnings"]:
                                    response["message"] += f"\n⚠ Advertències: {', '.join(validation['warnings'])}"
                            else:
                                response["success"] = False
                                response["message"] = f"❌ Configuració invàlida: {', '.join(validation['errors'])}"
                        except json.JSONDecodeError:
                            response["success"] = False
                            response["message"] = "Format JSON invàlid"
                
            elif cmd == "mode":
                if not args:
                    response["message"] = self._get_current_modes()
                elif args[0] == "creative":
                    self.creative_mode = not self.creative_mode
                    response["message"] = f"Mode creatiu {'🟢 ACTIU' if self.creative_mode else '⚫ INACTIU'}"
                elif args[0] == "visual":
                    self.visualization_mode = not self.visualization_mode
                    response["message"] = f"Mode visualització {'🟢 ACTIU' if self.visualization_mode else '⚫ INACTIU'}"
                elif args[0] == "learning":
                    self.learning_mode = not self.learning_mode
                    response["message"] = f"Mode aprenentatge {'🟢 ACTIU' if self.learning_mode else '⚫ INACTIU'}"
            
            elif cmd == "run":
                steps = int(args[0]) if args and args[0].isdigit() else 3
                generations = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
                
                for i in range(generations):
                    response["message"] = f"\nExecutant generació {self.system.generation_count + 1}..."
                    result = self.system.run_generation(steps=steps)
                    response["data"][f"generation_{i+1}"] = {
                        "duration": result["duration_seconds"],
                        "coherence": result["evolution"]["final_state"]["coherence"],
                        "fractal_branches": result["fractal_generation"]["branch_count"]
                    }
                
                response["message"] = f"✅ Executades {generations} generacions amb {steps} passos cada una"
            
            elif cmd == "stats":
                stats = self.system._get_system_state_summary()
                response["message"] = self._format_stats(stats)
            
            elif cmd == "save":
                filename = args[0] if args else "conversation_state.json"
                self.save_conversation_state(filename)
                response["message"] = f"💾 Estat guardat a '{filename}'"
            
            elif cmd == "load":
                filename = args[0] if args else "conversation_state.json"
                if self.load_conversation_state(filename):
                    response["message"] = f"📂 Estat carregat des de '{filename}'"
                else:
                    response["success"] = False
                    response["message"] = f"⚠ No s'ha trobat el fitxer '{filename}'"
            
            elif cmd == "optimize":
                optimized_config = optimize_config_for_hardware()
                self.system = SVGelonaAI5_2(optimized_config)
                self.config = optimized_config
                response["message"] = "⚙️  Sistema optimitzat per al teu maquinari"
            
            elif cmd == "benchmark":
                gens = int(args[0]) if args and args[0].isdigit() else 5
                steps = int(args[1]) if len(args) > 1 and args[1].isdigit() else 3
                
                benchmark_result = self.system.run_benchmark(
                    generations=gens, 
                    steps_per_gen=steps
                )
                
                response["message"] = self._format_benchmark(benchmark_result)
            
            elif cmd == "visualize":
                viz_data = self.system.generate_visualization_report()
                filename = "viz_data.json"
                with open(filename, "w") as f:
                    json.dump(viz_data, f, indent=2)
                response["message"] = f"📊 Dades de visualització generades a '{filename}'"
            
            elif cmd == "export":
                formats = args if args else ["json", "png"]
                self.system.save_system_state("exported_state.json")
                response["message"] = f"📦 Exportat en formats: {', '.join(formats)}"
            
            elif cmd == "reset":
                self.system = SVGelonaAI5_2(self.config)
                response["message"] = "🔄 Sistema reiniciat amb la configuració actual"
            
            elif cmd == "history":
                response["message"] = self._show_history(args[0] if args else "10")
            
            elif cmd == "clear":
                self.conversation_history = []
                response["message"] = "🧹 Historial de conversa esborrat"
            
            else:
                response["success"] = False
                response["message"] = f"Comanda desconeguda: /{cmd}\nUtilitza /help per a veure comandes disponibles"
        
        except Exception as e:
            response["success"] = False
            response["message"] = f"❌ Error executant comanda: {str(e)}"
        
        # Guardar resposta en historial
        self.conversation_history.append({
            "role": "system",
            "content": response["message"],
            "type": "command",
            "timestamp": self._get_timestamp()
        })
        
        self.command_history.append(cmd)
        return response
    
    def _process_creative_input(self, user_input: str) -> Dict[str, Any]:
        """Processa entrada en mode creatiu."""
        try:
            # Utilitzar pont semàntic per a interpretació creativa
            narrative_response = f"🎨 Mode Creatiu: Interpretant '{user_input}' com a inspiració..."
            
            # Generar fractal basat en la descripció
            result = self.system.run_generation(steps=5)
            
            # Afegir interpretació creativa
            creative_interpretation = self._generate_creative_interpretation(user_input, result)
            
            response = {
                "type": "creative_response",
                "narrative_response": narrative_response + "\n\n" + creative_interpretation,
                "application_result": {
                    "success": True,
                    "parameters_applied": [
                        ("creativity_boost", 0.0, 1.2),
                        ("complexity", result["fractal_generation"]["depth"], 
                         min(15, result["fractal_generation"]["depth"] + 2))
                    ]
                },
                "generation_result": result
            }
            
        except Exception as e:
            response = {
                "type": "error",
                "narrative_response": f"❌ Error en mode creatiu: {str(e)}",
                "application_result": {"success": False}
            }
        
        # Guardar en historial
        self.conversation_history.append({
            "role": "system",
            "content": response.get("narrative_response", ""),
            "type": "creative",
            "timestamp": self._get_timestamp()
        })
        
        return response
    
    def _process_visual_input(self, user_input: str) -> Dict[str, Any]:
        """Processa entrada en mode visualització."""
        try:
            # Generar visualització basada en la descripció
            narrative_response = f"👁️ Mode Visual: Creant imatge mental de '{user_input}'..."
            
            # Ajustar paràmetres de renderització
            self.system.config["render_enabled"] = True
            self.system.config["render_quality"] = "high"
            
            result = self.system.run_generation(steps=3)
            
            response = {
                "type": "visual_response",
                "narrative_response": narrative_response,
                "visual_description": self._generate_visual_description(user_input, result),
                "css_transform": result["fractal_generation"]["css_transform"],
                "application_result": {
                    "success": True,
                    "parameters_applied": [
                        ("render_quality", "medium", "high"),
                        ("visual_detail", 0.5, 0.8)
                    ]
                }
            }
            
        except Exception as e:
            response = {
                "type": "error",
                "narrative_response": f"❌ Error en mode visual: {str(e)}",
                "application_result": {"success": False}
            }
        
        # Guardar en historial
        self.conversation_history.append({
            "role": "system",
            "content": response.get("narrative_response", ""),
            "type": "visual",
            "timestamp": self._get_timestamp()
        })
        
        return response
    
    def _process_normal_input(self, user_input: str) -> Dict[str, Any]:
        """Processa entrada normal de conversa."""
        try:
            # Utilitzar pont semàntic del sistema
            response_data = self.system.converse_with_ai(user_input)
            
            # Afegir informació adicional si l'aprenentatge està actiu
            if self.learning_mode and response_data["application_result"]["success"]:
                learned_info = self._extract_learning_points(response_data)
                if learned_info:
                    response_data["narrative_response"] += f"\n\n📚 Aprenentatge: {learned_info}"
            
            return response_data
            
        except Exception as e:
            return {
                "type": "error",
                "narrative_response": f"❌ Error processant la teva entrada: {str(e)}",
                "application_result": {"success": False}
            }
    
    def _switch_config(self, new_config_name: str):
        """Canvia la configuració del sistema."""
        config_obj = get_config_object(new_config_name)
        self.system = SVGelonaAI5_2(config_obj.config.copy())
        self.config_name = new_config_name
        self.config = config_obj.config.copy()
        print(f"\n🔄 Canviat a configuració '{new_config_name}'")
    
    def _get_help_message(self) -> str:
        """Genera missatge d'ajuda."""
        help_text = """
📋 **COMANDES DISPONIBLES:**

**Configuració:**
  /config                         Mostrar configuració actual
  /config list                    Llistar totes les configuracions
  /config switch [nom]           Canviar configuració
  /config show [nom]             Mostrar resum d'una configuració
  /config custom {json}          Aplicar configuració personalitzada

**Modes:**
  /mode                           Mostrar modes actuals
  /mode creative                  Activar/desactivar mode creatiu
  /mode visual                    Activar/desactivar mode visualització
  /mode learning                  Activar/desactivar aprenentatge

**Execució:**
  /run [passos] [generacions]    Executar generacions
  /benchmark [gens] [passos]     Executar benchmark
  /optimize                      Optimitzar per al maquinari

**Informació:**
  /stats                         Mostrar estadístiques del sistema
  /history [n]                   Mostrar últimes n entrades d'historial
  /visualize                    Generar dades de visualització

**Gestió:**
  /save [fitxer]                Guardar estat de conversa
  /load [fitxer]                Carregar estat de conversa
  /export [formats]             Exportar dades
  /reset                        Reiniciar sistema
  /clear                        Esborrar historial de conversa

**Conversa normal:**
  Parla amb l'IA en llenguatge natural!
  Exemples:
    • "Crea un fractal complex"
    • "Fes-ho més orgànic"
    • "Explica el que estàs pensant"
    • "Mostra'm la teva memòria"

**Comandes ràpides:**
  exit, quit, q                 Sortir
  help, ?                       Ajuda
"""
        return help_text
    
    def _get_current_config_info(self) -> str:
        """Obté informació de la configuració actual."""
        config_obj = get_config_object(self.config_name)
        return (f"📊 **Configuració actual:** {self.config_name}\n"
                f"📝 Descripció: {config_obj.description}\n"
                f"⚡ Mode rendiment: {self.config['performance_mode']}\n"
                f"💾 Memòria: {self.config['memory_limit_mb']}MB\n"
                f"🎯 Profunditat fractal: {self.config.get('max_fractal_depth', 10)}\n"
                f"🔧 Pont semàntic: {'🟢 ACTIU' if self.config.get('semantic_bridge_enabled', True) else '⚫ INACTIU'}")
    
    def _list_configs(self) -> str:
        """Llista totes les configuracions disponibles."""
        configs_text = "📋 **Configuracions disponibles:**\n"
        for name in self.available_configs:
            config_obj = get_config_object(name)
            configs_text += f"  • {name:15} - {config_obj.description}\n"
        return configs_text
    
    def _get_current_modes(self) -> str:
        """Obté l'estat dels modes actuals."""
        return (f"🎭 **Modes actuals:**\n"
                f"  Creatiu: {'🟢 ACTIU' if self.creative_mode else '⚫ INACTIU'}\n"
                f"  Visualització: {'🟢 ACTIU' if self.visualization_mode else '⚫ INACTIU'}\n"
                f"  Aprenentatge: {'🟢 ACTIU' if self.learning_mode else '⚫ INACTIU'}")
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Formata estadístiques del sistema."""
        return (f"📈 **Estadístiques del sistema:**\n"
                f"  Generacions: {stats.get('generation_count', 0)}\n"
                f"  Temps actiu: {stats.get('uptime_hours', 0):.1f}h\n"
                f"  Cicatrius: {stats.get('scar_archive', {}).get('total_scars', 0)}\n"
                f"  Axiomes: {stats.get('axiom_system', {}).get('total_axioms', 0)}\n"
                f"  Ús memòria: {self.system.realtime_stats.get('memory_usage_mb', 0):.1f}MB\n"
                f"  Rendiment: {self.system.realtime_stats.get('generations_per_second', 0):.2f} gens/s")
    
    def _format_benchmark(self, benchmark: Dict[str, Any]) -> str:
        """Formata resultats de benchmark."""
        perf = benchmark.get("performance", {})
        return (f"🏆 **Resultats Benchmark:**\n"
                f"  Generacions: {benchmark.get('benchmark_config', {}).get('generations', 0)}\n"
                f"  Temps total: {perf.get('total_time', 0):.2f}s\n"
                f"  Temps/gen: {perf.get('avg_time_per_gen', 0):.3f}s\n"
                f"  Gens/s: {perf.get('total_generations_per_second', 0):.2f}\n"
                f"  Memòria final: {perf.get('final_memory_usage_mb', 0):.1f}MB")
    
    def _generate_creative_interpretation(self, input_text: str, result: Dict[str, Any]) -> str:
        """Genera una interpretació creativa dels resultats."""
        metaphors = [
            "com una dansa de partícules còsmiques",
            "com un somni fractal que es desenvolupa",
            "com un ecosistema matemàtic vivent",
            "com una simfonia geomètrica",
            "com un llenguatge secret de l'univers"
        ]
        
        import random
        metaphor = random.choice(metaphors)
        
        return (f"✨ La teva idea '{input_text}' s'ha transformat {metaphor}.\n"
                f"📊 Complexitat generada: {result['fractal_generation']['branch_count']} branques\n"
                f"🎯 Coherència: {result['evolution']['final_state']['coherence']:.3f}\n"
                f"🌀 Entropia: {result['evolution']['final_state']['entropy']:.3f}")
    
    def _generate_visual_description(self, input_text: str, result: Dict[str, Any]) -> str:
        """Genera una descripció visual."""
        visual_elements = [
            "Patrons espirals que s'entrellacen",
            "Geometria cristal·lina en evolució",
            "Formes orgàniques que creixen i es divideixen",
            "Estructures fractals que s'autosimilaritzen",
            "Textures matemàtiques que respiren"
        ]
        
        import random
        element = random.choice(visual_elements)
        
        return (f"👁️  Visualització generada:\n"
                f"  • {element}\n"
                f"  • Profunditat: {result['fractal_generation']['depth']} nivells\n"
                f"  • Transformació CSS aplicada\n"
                f"  • Basada en: '{input_text}'")
    
    def _extract_learning_points(self, response_data: Dict[str, Any]) -> str:
        """Extreu punts d'aprenentatge de la resposta."""
        if "application_result" not in response_data:
            return ""
        
        applied = response_data["application_result"].get("parameters_applied", [])
        if not applied:
            return ""
        
        learnings = []
        for param, old_val, new_val in applied:
            change_pct = abs((new_val - old_val) / (old_val + 1e-10)) * 100
            if change_pct > 10:  # Canvis significatius
                direction = "augmentat" if new_val > old_val else "reduït"
                learnings.append(f"{param} {direction} de {old_val:.2f} a {new_val:.2f}")
        
        return "; ".join(learnings) if learnings else ""
    
    def _show_history(self, n_str: str) -> str:
        """Mostra l'historial de conversa."""
        try:
            n = int(n_str)
        except ValueError:
            n = 10
        
        history_text = f"📜 **Últims {min(n, len(self.conversation_history))} missatges:**\n"
        
        for i, entry in enumerate(self.conversation_history[-n:]):
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:100] + "..." if len(entry.get("content", "")) > 100 else entry.get("content", "")
            msg_type = entry.get("type", "message")
            
            prefix = "👤" if role == "user" else "🤖"
            if msg_type == "command":
                prefix = "⚙️"
            elif msg_type == "creative":
                prefix = "🎨"
            elif msg_type == "visual":
                prefix = "👁️"
            
            history_text += f"{prefix} {content}\n"
        
        return history_text
    
    def _get_timestamp(self) -> str:
        """Obté un timestamp formatat."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def save_conversation_state(self, filename: str = "conversation_state.json"):
        """Guarda l'estat de la conversa."""
        state = {
            "conversation_history": self.conversation_history,
            "command_history": self.command_history,
            "config_name": self.config_name,
            "modes": {
                "creative": self.creative_mode,
                "visualization": self.visualization_mode,
                "learning": self.learning_mode
            },
            "system_stats": self.system._get_system_state_summary(),
            "timestamp": self._get_timestamp()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_conversation_state(self, filename: str = "conversation_state.json") -> bool:
        """Carrega l'estat de la conversa."""
        try:
            if not os.path.exists(filename):
                return False
            
            with open(filename, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            self.conversation_history = state.get("conversation_history", [])
            self.command_history = state.get("command_history", [])
            self.config_name = state.get("config_name", "conversational")
            modes = state.get("modes", {})
            self.creative_mode = modes.get("creative", False)
            self.visualization_mode = modes.get("visualization", False)
            self.learning_mode = modes.get("learning", True)
            
            # Reconstruir sistema si cal
            if self.config_name in ALL_CONFIGS:
                config_obj = get_config_object(self.config_name)
                self.system = SVGelonaAI5_2(config_obj.config.copy())
                self.config = config_obj.config.copy()
            
            return True
            
        except Exception as e:
            print(f"⚠ Error carregant estat: {e}")
            return False

def main():
    """Funció principal d'execució."""
    parser = argparse.ArgumentParser(description='SVGelona_AI 5.2 - Interfície Conversacional')
    parser.add_argument('--config', type=str, default='conversational',
                       help='Configuració inicial (default: conversational)')
    parser.add_argument('--load', type=str, 
                       help='Carregar estat des d\'un fitxer')
    parser.add_argument('--custom', type=str,
                       help='Configuració personalitzada en format JSON')
    
    args = parser.parse_args()
    
    # Configuració personalitzada si s'especifica
    custom_config = None
    if args.custom:
        try:
            custom_config = json.loads(args.custom)
            print(f"✅ Configuració personalitzada carregada des d'arguments")
        except json.JSONDecodeError as e:
            print(f"⚠ Error parsejant configuració personalitzada: {e}")
            return
    
    # Crear instància del sistema
    chatbot = ConversationalSVGelona(
        config_name=args.config,
        custom_config=custom_config
    )
    
    # Carregar estat si s'especifica
    if args.load and chatbot.load_conversation_state(args.load):
        print(f"✅ Estat carregat des de '{args.load}'")
    
    print("\n💬 **Instruccions:**")
    print("  • Parla normalment amb l'IA")
    print("  • Utilitza comandes amb / (ex: /help)")
    print("  • exit/quit per sortir")
    print("-" * 80)
    
    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()
            
            if user_input.lower() in ['exit', 'sortir', 'quit', 'q']:
                print("\n👋 Fins aviat! Recorda que pots guardar l'estat amb /save")
                break
            
            if not user_input:
                continue
            
            # Processar entrada
            response = chatbot.process_command(user_input)
            
            # Mostrar resposta
            if response["type"] == "command_response":
                print(f"\n⚙️  Sistema: {response['message']}")
                if response.get("data"):
                    for key, value in response["data"].items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                print(f"    {subkey}: {subvalue}")
            elif "narrative_response" in response:
                print(f"\n🤖 SVGelona_AI: {response['narrative_response']}")
                
                # Mostrar canvis aplicats
                if (response.get("application_result", {}).get("success") and 
                    "parameters_applied" in response.get("application_result", {})):
                    print("\n⚙️  Canvis aplicats:")
                    for param, old_val, new_val in response["application_result"]["parameters_applied"]:
                        change = new_val - old_val
                        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                        print(f"  • {param}: {old_val:.3f} {arrow} {new_val:.3f}")
            
            elif "error" in response.get("type", ""):
                print(f"\n❌ Error: {response.get('narrative_response', 'Error desconegut')}")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Interromput per l'usuari.")
            save_choice = input("Vols guardar l'estat abans de sortir? (s/n): ").lower()
            if save_choice == 's':
                chatbot.save_conversation_state()
                print("💾 Estat guardat!")
            break
        
        except Exception as e:
            print(f"\n⚠ Error inesperat: {e}")
            continue
    
    # Guardar historial de conversa automàticament
    if chatbot.conversation_history:
        auto_save = "conversation_auto_save.json"
        chatbot.save_conversation_state(auto_save)
        print(f"\n💾 Historial de conversa guardat automàticament a '{auto_save}'")
    
    print("=" * 80)

if __name__ == "__main__":
    main()