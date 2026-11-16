import 'package:flutter/material.dart';

import '../../widgets/section_card.dart';

/// Records only the instructions the user can read on the actual garment label.
class CareLabelScreen extends StatefulWidget {
  final Map<String, dynamic> instructions;
  const CareLabelScreen({super.key, required this.instructions});

  @override
  State<CareLabelScreen> createState() => _CareLabelScreenState();
}

class _CareLabelScreenState extends State<CareLabelScreen> {
  final _form = GlobalKey<FormState>();
  late final _instructions = Map<String, dynamic>.of(widget.instructions);
  late final _temperature = TextEditingController(
      text: _instructions['max_temperature_c']?.toString());

  bool get _canWash => const {
        'hand',
        'machine_normal',
        'machine_mild',
        'machine_very_mild'
      }.contains(_instructions['wash']);

  @override
  void dispose() {
    _temperature.dispose();
    super.dispose();
  }

  Widget _field(String key, String title, Map<String, String> options) =>
      Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: DropdownButtonFormField<String>(
          initialValue: _instructions[key] as String? ?? 'unknown',
          isExpanded: true,
          decoration: InputDecoration(labelText: title),
          items: [
            const DropdownMenuItem(value: 'unknown', child: Text('Unknown')),
            for (final entry in options.entries)
              DropdownMenuItem(value: entry.key, child: Text(entry.value)),
          ],
          onChanged: (value) => setState(() => _instructions[key] = value),
        ),
      );

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Garment care label')),
        body: SafeArea(
          child: Form(
            key: _form,
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                SectionCard(
                  title: 'Enter label instructions',
                  icon: Icons.sell_outlined,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Copy the instructions you can verify on the garment. '
                        'Leave unclear or unlisted instructions as Unknown.',
                      ),
                      const SizedBox(height: 16),
                      _field('wash', 'Washing', const {
                        'dry_clean_only': 'Dry clean only',
                        'professional_only': 'Professional care only',
                        'do_not_wash': 'Do not wash',
                        'hand': 'Hand wash only',
                        'machine_normal': 'Normal machine wash',
                        'machine_mild': 'Mild machine wash',
                        'machine_very_mild': 'Very mild machine wash',
                        'conflicting': 'Conflicting washing instructions',
                      }),
                      if (_canWash)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: TextFormField(
                            controller: _temperature,
                            keyboardType: TextInputType.number,
                            decoration: const InputDecoration(
                                labelText: 'Label maximum in °C (optional)'),
                            validator: (text) {
                              final value = text?.trim() ?? '';
                              if (value.isEmpty) return null;
                              final temperature = int.tryParse(value);
                              return temperature != null &&
                                      temperature >= 1 &&
                                      temperature <= 95
                                  ? null
                                  : 'Enter a whole number from 1 to 95.';
                            },
                          ),
                        ),
                      _field('dry', 'Drying', const {
                        'do_not_tumble': 'Do not tumble dry',
                        'line': 'Line dry',
                        'flat': 'Dry flat',
                        'tumble_low': 'Tumble dry low',
                        'tumble_normal': 'Tumble dry normal',
                      }),
                      _field('iron', 'Ironing', const {
                        'do_not_iron': 'Do not iron',
                        'low': 'Low heat',
                        'medium': 'Medium heat',
                        'high': 'High heat',
                      }),
                      _field('bleach', 'Bleaching', const {
                        'do_not_bleach': 'Do not bleach',
                        'oxygen_only': 'Oxygen bleach only',
                        'allowed': 'Bleaching permitted',
                      }),
                      _field('professional', 'Professional care', const {
                        'do_not_dry_clean': 'Do not dry clean',
                      }),
                      FilledButton(
                        onPressed: () {
                          if (!_form.currentState!.validate()) return;
                          Navigator.of(context).pop({
                            ..._instructions,
                            'max_temperature_c': _canWash
                                ? int.tryParse(_temperature.text.trim())
                                : null,
                          });
                        },
                        child: const Text('Use these instructions'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      );
}
