#!/usr/bin/env python3
"""
Script d'execució optimitzat per a SVGelona_AI 5.2
"""
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='SVGelona_AI 5.2 - Sistema d\'IA fractal optimitzat')
    
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'benchmark', 'generate', 'optimize'],
                       help='Mode d\'execució')
    
    parser.add_argument('--generations', type=int, default=10,
                       help='Nombre de generacions a executar')
    
    parser.add_argument('--steps', type=int, default=5,
                       help='Passos per generació')
    
    parser.add_argument('--memory', type=int, default=100,
                       help='Límit de memòria en MB')
    
    parser.add_argument('--depth', type=int, default=12,
                       help='Profunditat fractal màxima')
    
    parser.add_argument('--performance', type=str, default='balanced',
                       choices=['performance', 'balanced', 'quality'],
                       help='Mode de rendiment')
    
    parser.add_argument('--output', type=str, default='output',
                       help='Directori de sortida')
    
    args = parser.parse_args()
    
    # Configuració basada en arguments
    config = {
        "max_scars": 10000,
        "max_fractal_depth": args.depth,
        "memory_limit_mb": args.memory,
        "performance_mode": args.performance,
        "auto_optimize": True,
        "save_state_interval": 100,
        "render_enabled": True
    }
    
    # Importar sistema
    from main_v5_2 import SVGelonaAI5_2
    
    print(f"SVGelona_AI 5.2 - Mode: {args.mode}")
    print(f"Configuració: {config}")
    print()
    
    # Crear sistema
    system = SVGelonaAI5_2(config)
    
    # Executar segons mode
    if args.mode == 'benchmark':
        print(f"Executant benchmark de {args.generations} generacions...")
        results = system.run_benchmark(
            generations=args.generations,
            steps_per_gen=args.steps
        )
        
        # Guardar resultats
        import json
        with open(os.path.join(args.output, 'benchmark.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark completat. Resultats guardats a {args.output}/benchmark.json")
    
    elif args.mode == 'generate':
        print(f"Generant {args.generations} generacions...")
        
        for i in range(args.generations):
            print(f"\nGeneració {i+1}/{args.generations}")
            system.run_generation(steps=args.steps)
        
        # Guardar estat
        system.save_system_state(
            os.path.join(args.output, 'svgelona_final_state.json')
        )
        
        print(f"\nGeneració completada. Estat guardat a {args.output}/")
    
    elif args.mode == 'optimize':
        print("Mode d'optimització - Executant amb ajustos agressius...")
        
        # Configuració agressiva d'optimització
        config["performance_mode"] = "performance"
        config["memory_limit_mb"] = args.memory // 2  # Ús més conservador
        
        system = SVGelonaAI5_2(config)
        
        # Executar i optimitzar
        for i in range(args.generations):
            result = system.run_generation(steps=args.steps, optimize=True)
            
            # Mostrar progrés d'optimització
            if "optimization" in result and result["optimization"]["optimizations_applied"] > 0:
                print(f"  Generació {i+1}: {result['optimization']['optimizations_applied']} optimitzacions aplicades")
        
        print(f"\nOptimització completada. {system.generation_count} generacions processades.")
    
    else:  # interactive
        print("Mode interactiu - Escriu 'help' per a veure comandes")
        print()
        
        while True:
            try:
                command = input("svgelona> ").strip().lower()
                
                if command == 'help':
                    print("\nComandes disponibles:")
                    print("  run [n] - Executar n generacions (per defecte: 1)")
                    print("  benchmark [g] [s] - Executar benchmark (g generacions, s passos)")
                    print("  stats - Mostrar estadístiques del sistema")
                    print("  save [file] - Guardar estat del sistema")
                    print("  visualize - Generar dades de visualització")
                    print("  optimize - Executar optimització de memòria")
                    print("  exit - Sortir")
                    print()
                
                elif command.startswith('run'):
                    parts = command.split()
                    n = int(parts[1]) if len(parts) > 1 else 1
                    
                    for i in range(n):
                        print(f"\nExecutant generació {system.generation_count + 1}...")
                        system.run_generation(steps=3)
                
                elif command.startswith('benchmark'):
                    parts = command.split()
                    g = int(parts[1]) if len(parts) > 1 else 5
                    s = int(parts[2]) if len(parts) > 2 else 3
                    
                    results = system.run_benchmark(generations=g, steps_per_gen=s)
                    print(f"Benchmark completat. Temps total: {results['performance']['total_time']:.2f}s")
                
                elif command == 'stats':
                    stats = system._get_system_state_summary()
                    print(f"\nGeneracions: {stats['generation_count']}")
                    print(f"Cicatrius: {stats['scar_archive']['total_scars']}")
                    print(f"Memòria: {system.realtime_stats['memory_usage_mb']:.1f} MB")
                    print(f"Rendiment: {system.realtime_stats['generations_per_second']:.2f} gens/s")
                    print(f"Coherència actual: {system.engine.state.coherence:.3f}")
                    print(f"Entropia actual: {system.engine.state.entropy:.3f}")
                
                elif command.startswith('save'):
                    parts = command.split()
                    filename = parts[1] if len(parts) > 1 else "svgelona_state.json"
                    
                    if system.save_system_state(filename):
                        print(f"Estat guardat a {filename}")
                    else:
                        print("Error guardant estat")
                
                elif command == 'visualize':
                    report = system.generate_visualization_report()
                    print(f"Dades de visualització generades")
                    print(f"  Generacions en històric: {len(report['time_series']['generations'])}")
                    print(f"  Fases angulars: {report['phase_distribution']}")
                
                elif command == 'optimize':
                    print("Executant optimització de memòria...")
                    result = system.memory_manager.perform_memory_management()
                    
                    if "items_affected" in result:
                        evicted = result["items_affected"].get("evicted_count", 0)
                        print(f"  {evicted} cicatrius eliminades")
                        print(f"  Pressió de memòria: {result['final_memory_pressure']:.2f}")
                
                elif command in ['exit', 'quit', 'q']:
                    print("Sortint...")
                    break
                
                else:
                    print("Comanda no reconeguda. Escriu 'help' per a ajuda.")
            
            except KeyboardInterrupt:
                print("\nInterromput per l'usuari.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nSVGelona_AI 5.2 - Execució finalitzada")

if __name__ == "__main__":
    main()