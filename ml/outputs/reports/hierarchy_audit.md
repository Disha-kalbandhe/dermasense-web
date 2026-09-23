# Reconciled Label and Hierarchy Audit

- Raw train labels: 233
- Raw union labels: 245
- Production classes: 106
- Rule: keep labels with at least 3 train samples and at least 1 test sample.
- Excluded labels: 139
- Mainclass count: 8
- Subclass count: 19

## Manual reviews

- Reapplied: Fissure Foot, Keratoderma, Lichen Planus pigmentosus, Lichen Simplex Chronicus, Molluscum Contagiosum, Palmoplantar Keratoderma, Pustular Scabies, Scabies Infected, Sebaceous Cyst
- Previously reviewed but excluded: Diabetic Ulcer, Paronychia, Pseudomonas, Tinea Infected with secondary bacterial Infection, Zoon's Balanitis

## Exclusions

| Label | Train | Test | Reasons |
|---|---:|---:|---|
| Acne Keloidalis | 1 | 2 | insufficient_train_samples |
| Anetoderma | 2 | 0 | insufficient_train_samples, missing_from_test |
| Angioedema | 1 | 1 | insufficient_train_samples |
| Balanitis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Becker nevus | 1 | 0 | insufficient_train_samples, missing_from_test |
| Bowen's Metastasis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Bullous Fixed Drug Eruption | 0 | 1 | insufficient_train_samples |
| Bullous Pemphigoid | 2 | 1 | insufficient_train_samples |
| Cellulitis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Chromoblastomycosis | 8 | 0 | missing_from_test |
| Chronic Bullous Disease of Childhood | 8 | 0 | missing_from_test |
| Chronic eczema with secondary infection | 1 | 0 | insufficient_train_samples, missing_from_test |
| Congenital Disease | 2 | 0 | insufficient_train_samples, missing_from_test |
| Contact Dermatitis with secondary infection | 2 | 0 | insufficient_train_samples, missing_from_test |
| Contact purpura | 1 | 1 | insufficient_train_samples |
| Corn | 2 | 0 | insufficient_train_samples, missing_from_test |
| Crusted eczematous dermatitis | 2 | 0 | insufficient_train_samples, missing_from_test |
| Cutaneous Anthrax | 1 | 0 | insufficient_train_samples, missing_from_test |
| Cutaneous amyloidosis | 2 | 4 | insufficient_train_samples |
| Cutaneous larva migrans | 1 | 0 | insufficient_train_samples, missing_from_test |
| Cutaneous sarcoidosis | 3 | 0 | missing_from_test |
| Cyst | 0 | 1 | insufficient_train_samples |
| Darier's disease | 5 | 0 | missing_from_test |
| Demodicosis | 0 | 3 | insufficient_train_samples |
| Dermatofibroma | 0 | 1 | insufficient_train_samples |
| Dermatosis Papulosa Nigra | 17 | 0 | missing_from_test |
| Diabetic Ulcer | 9 | 0 | missing_from_test |
| Discoid Lupus Erythematosus | 12 | 0 | missing_from_test |
| Drug Rash | 3 | 0 | missing_from_test |
| Drug eruption | 5 | 0 | missing_from_test |
| Dry Discoid Eczema | 3 | 0 | missing_from_test |
| Dryness (xerosis) | 7 | 0 | missing_from_test |
| Ecchymosis | 1 | 1 | insufficient_train_samples |
| Ecthyma | 2 | 2 | insufficient_train_samples |
| Eczemated Tinea | 1 | 0 | insufficient_train_samples, missing_from_test |
| Epidermodysplasia Verruciformis | 2 | 0 | insufficient_train_samples, missing_from_test |
| Epidermolysis Bullosa | 7 | 0 | missing_from_test |
| Erythema Annulare Centrifugum | 2 | 1 | insufficient_train_samples |
| Erythema multiforme | 2 | 0 | insufficient_train_samples, missing_from_test |
| Erythrasma | 1 | 0 | insufficient_train_samples, missing_from_test |
| Facial hypermelanosis | 3 | 0 | missing_from_test |
| Foot ulcer | 2 | 0 | insufficient_train_samples, missing_from_test |
| GeographicTongue | 2 | 0 | insufficient_train_samples, missing_from_test |
| Granuloma annulare | 2 | 0 | insufficient_train_samples, missing_from_test |
| Gutted Lichen Planus | 1 | 0 | insufficient_train_samples, missing_from_test |
| Haemorrhoids | 1 | 0 | insufficient_train_samples, missing_from_test |
| Herpes Labialis | 1 | 2 | insufficient_train_samples |
| Hyper Hydrosis | 0 | 1 | insufficient_train_samples |
| Hyperpigmentation disorder | 1 | 0 | insufficient_train_samples, missing_from_test |
| Hypertrophic Scar / Keloid | 12 | 0 | missing_from_test |
| Infected Keratoderma | 6 | 0 | missing_from_test |
| Infected Nodule | 1 | 0 | insufficient_train_samples, missing_from_test |
| Infection of toe web | 1 | 0 | insufficient_train_samples, missing_from_test |
| Inflammatory Linear Verrucous Epidermal Nevus | 1 | 0 | insufficient_train_samples, missing_from_test |
| Inverse psoriasis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Itchy skin eruption | 1 | 1 | insufficient_train_samples |
| Keloid Milium | 5 | 0 | missing_from_test |
| Keratoacanthoma | 1 | 0 | insufficient_train_samples, missing_from_test |
| Keratoderma with Secondary Infection | 0 | 1 | insufficient_train_samples |
| Keratolysis Exfoliativa | 3 | 0 | missing_from_test |
| Kerion | 0 | 1 | insufficient_train_samples |
| Kumkum Dermatitis | 4 | 0 | missing_from_test |
| Kyrle���������s Disease | 1 | 0 | insufficient_train_samples, missing_from_test |
| Lichen sclerosus | 6 | 0 | missing_from_test |
| Lichen simplex | 1 | 0 | insufficient_train_samples, missing_from_test |
| Lichen spinulosus | 1 | 1 | insufficient_train_samples |
| Lichen striatus | 3 | 0 | missing_from_test |
| Lichenoid eruption | 4 | 0 | missing_from_test |
| Linear Lichen Planus | 1 | 1 | insufficient_train_samples |
| Lipoma | 2 | 3 | insufficient_train_samples |
| Maduromycosis | 3 | 0 | missing_from_test |
| Melanoma | 2 | 0 | insufficient_train_samples, missing_from_test |
| Miliaria | 1 | 0 | insufficient_train_samples, missing_from_test |
| Mongolian spot | 1 | 0 | insufficient_train_samples, missing_from_test |
| Morphea | 1 | 0 | insufficient_train_samples, missing_from_test |
| Mucosal Ulcer or Cyst | 2 | 1 | insufficient_train_samples |
| Mucus Cyst | 1 | 0 | insufficient_train_samples, missing_from_test |
| Myxoid Cyst | 1 | 0 | insufficient_train_samples, missing_from_test |
| Nail dystrophy | 2 | 0 | insufficient_train_samples, missing_from_test |
| Necrotizing Fasciitis | 2 | 0 | insufficient_train_samples, missing_from_test |
| Neurofibroma | 1 | 1 | insufficient_train_samples |
| Nevus Depigmentosis | 2 | 0 | insufficient_train_samples, missing_from_test |
| Nevus anemicus | 2 | 0 | insufficient_train_samples, missing_from_test |
| Nevus sebaceous | 0 | 1 | insufficient_train_samples |
| Nodular vasculitis | 0 | 1 | insufficient_train_samples |
| Nutritional Dermatitis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Ochronosis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Ophiasis | 2 | 0 | insufficient_train_samples, missing_from_test |
| Oral Candidiasis | 5 | 0 | missing_from_test |
| Paederus dermatitis | 7 | 0 | missing_from_test |
| Palmar psoriasis | 3 | 0 | missing_from_test |
| Paronychia | 3 | 0 | missing_from_test |
| Pellagra dermatitis | 7 | 0 | missing_from_test |
| Pemphigus foliaceus | 1 | 0 | insufficient_train_samples, missing_from_test |
| Perioral dermatitis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Phimosis | 4 | 0 | missing_from_test |
| Phrynoderma | 1 | 4 | insufficient_train_samples |
| Pigmented purpuric eruption | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pityriasis lichenoides chronica | 8 | 0 | missing_from_test |
| Pityriasis rosea | 9 | 0 | missing_from_test |
| Port wine stain | 8 | 0 | missing_from_test |
| Prurigo | 5 | 0 | missing_from_test |
| Pseudo lymphoma | 0 | 1 | insufficient_train_samples |
| Pseudoepitheliomatous keratotic & micaceous balanitis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pseudofolliculitis barbae | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pseudolymphoma | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pseudomonas | 2 | 0 | insufficient_train_samples, missing_from_test |
| Psoriasis Vulgaris | 8 | 0 | missing_from_test |
| Purpuric Dermatoses | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pustular psoriasis | 1 | 0 | insufficient_train_samples, missing_from_test |
| Pyoderma gangrenosum | 1 | 1 | insufficient_train_samples |
| Reticulate acropigmentation of Dohi | 1 | 2 | insufficient_train_samples |
| Rosacea | 3 | 0 | missing_from_test |
| SLE - Systemic lupus erythematosus-related syndrome | 0 | 1 | insufficient_train_samples |
| Scleroderma | 7 | 0 | missing_from_test |
| Seborrheic Melanosis | 3 | 0 | missing_from_test |
| Skin ulcer | 2 | 1 | insufficient_train_samples |
| Staphylococcal Scalded Skin Syndrome | 2 | 0 | insufficient_train_samples, missing_from_test |
| Steatocystoma Multiplex | 1 | 0 | insufficient_train_samples, missing_from_test |
| Steroid Induced Facies | 8 | 0 | missing_from_test |
| Sun damaged skin | 4 | 0 | missing_from_test |
| Sunburn | 4 | 0 | missing_from_test |
| Sweat Dermatitis | 10 | 0 | missing_from_test |
| Syphilis | 3 | 0 | missing_from_test |
| Syringoma | 1 | 0 | insufficient_train_samples, missing_from_test |
| Tick bite | 1 | 0 | insufficient_train_samples, missing_from_test |
| Tinea Infected with secondary bacterial Infection | 3 | 0 | missing_from_test |
| Trichoepithelioma | 5 | 0 | missing_from_test |
| Trophic Ulcer | 1 | 0 | insufficient_train_samples, missing_from_test |
| Ulcer | 4 | 0 | missing_from_test |
| Urticarial vasculitis | 4 | 0 | missing_from_test |
| Varicella Zoster | 5 | 0 | missing_from_test |
| Varicose vein | 3 | 0 | missing_from_test |
| Vasculitis | 12 | 0 | missing_from_test |
| White piedra | 0 | 1 | insufficient_train_samples |
| Xanthelasma Palpebrarum | 1 | 0 | insufficient_train_samples, missing_from_test |
| Xanthoma | 1 | 0 | insufficient_train_samples, missing_from_test |
| Xerosis vulgaris | 7 | 0 | missing_from_test |
| Zoon's Balanitis | 2 | 0 | insufficient_train_samples, missing_from_test |
