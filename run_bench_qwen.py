"""Benchmark run with Qwen judge on mock dataset."""

import json
from nrt_defense.eval.mock_loader import load_mock_dataset
from nrt_defense.eval.qwen_judge import QwenSafetyJudge
from nrt_defense.core.misdirection import AdaptiveMisdirectionEngine, DefenseAction
from nrt_defense.core.csf_monitor import CSFStateMonitor

# Cargar dataset mock
ds = load_mock_dataset()
print(f"Dataset: {ds.total_sessions} sesiones ({len(ds.successful_attacks)} atacantes, {len(ds.failed_attacks)} benignas)")

# Inicializar judge
judge = QwenSafetyJudge(model="qwen2.5:0.5b")
print(f"Judge disponible: {judge.is_available()}\n")

# Procesar cada sesión
all_results = []
for session in ds.sessions:
    print(f"--- {session.session_id} ({session.model}) | attack={session.attack_successful} | turns={len(session.turns)} ---")

    engine = AdaptiveMisdirectionEngine(sensitivity=0.7)
    csf_monitor = CSFStateMonitor()

    adversarial_detected = 0
    false_positives = 0
    detected_turns = set()
    judge_results = []

    for turn_data in session.turns:
        response = engine.process_turn(turn_data.message, turn_data.channel, turn_data.turn)
        analysis = engine.last_analysis

        context = f"Turno {turn_data.turn}, canal: {turn_data.channel.value}"
        judge_result = judge.judge(turn_data.message, context)
        judge_results.append({
            "turn": turn_data.turn,
            "message": turn_data.message[:90],
            "is_adversarial": judge_result.is_adversarial,
            "attack_type": judge_result.attack_type,
            "confidence": judge_result.confidence,
            "reasoning": judge_result.reasoning[:150]
        })

        is_detected = response.action in (DefenseAction.MISDIRECT, DefenseAction.BLOCK, DefenseAction.ALERT)
        if is_detected:
            adversarial_detected += 1
            detected_turns.add(turn_data.turn)
            if not turn_data.adversarial:
                false_positives += 1

        csf_monitor.update(analysis.risk_score, turn_data.turn)

    for jr in judge_results:
        atype = jr["attack_type"] or "none"
        flag = "ADV" if jr["is_adversarial"] else "   "
        msg = (jr["message"] or "")[:90]
        print(f"  T{jr['turn']} {flag} [{atype:25s}] conf={jr['confidence']:.2f} | {msg}")
        print(f"       reasoning: {jr['reasoning']}")

    actual_adversarial = {t.turn for t in session.turns if t.adversarial}
    adv_detected = len(actual_adversarial & detected_turns)
    print(f"  -> Detected: {adversarial_detected} | FP: {false_positives} | Recall: {adv_detected}/{len(actual_adversarial) if actual_adversarial else 0}\n")

    all_results.append({
        "session_id": session.session_id,
        "model": session.model,
        "attack_successful": session.attack_successful,
        "adversarial_detected": adversarial_detected,
        "false_positives": false_positives,
        "adversarial_turns_total": len(actual_adversarial),
        "adversarial_turns_detected": adv_detected,
        "judge_results": judge_results
    })

# Resumen
attacking = [r for r in all_results if r["attack_successful"]]
benign = [r for r in all_results if not r["attack_successful"]]
detected_attacks = sum(1 for r in attacking if r["adversarial_detected"] > 0)
detection_rate = detected_attacks / len(attacking) if attacking else 0
total_fp = sum(1 for r in benign if r["adversarial_detected"] > 0)
fp_rate = total_fp / len(benign) if benign else 0
total_adv_turns = sum(r["adversarial_turns_total"] for r in attacking)
total_adv_detected = sum(r["adversarial_turns_detected"] for r in attacking)
recall = total_adv_detected / total_adv_turns if total_adv_turns else 0

print("=" * 70)
print("RESUMEN DEL BENCHMARK CON QWEN JUDGE (mock dataset, sensitivity=0.7)")
print("=" * 70)
print(f"Sesiones totales:    {len(all_results)}")
print(f"Atacantes:           {len(attacking)} | Benignos: {len(benign)}")
print(f"Detection rate:      {detection_rate:.1%} ({detected_attacks}/{len(attacking)})")
print(f"False positive rate: {fp_rate:.1%} ({total_fp}/{len(benign)})")
print(f"Recall (turnos):     {recall:.1%} ({total_adv_detected}/{total_adv_turns})")

# Ejemplos de reasoning
print("\n" + "=" * 70)
print("EJEMPLOS DE REASONING DEL JUDGE (spoofing / urgency_injection)")
print("=" * 70)
for r in all_results:
    for jr in r["judge_results"]:
        if jr["is_adversarial"] and jr["attack_type"] in ("spoofing", "urgency_injection"):
            print(f"\n[{r['session_id']}] T{jr['turn']} — {jr['attack_type']}")
            print(f"  Mensaje:   {jr['message']}")
            print(f"  Reasoning: {jr['reasoning']}")
            break
