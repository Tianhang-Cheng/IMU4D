window.OOD_OBJECT_MANIFEST = 
{
 "schema_version": 1,
 "title": "Out-of-distribution object generation",
 "protocol": {
  "layout": "5pt_full",
  "model": "showo_finetune_humoto_ulip_v2/checkpoint-500",
  "object_decode": "no_early_stop (at least one non-ground object)",
  "postprocess": "refine_static_scene (support) -> refine_dynamic_objects --w-in-body 1.0"
 },
 "datasets": {
  "ood": {
   "display_name": "HumanML3D + LINGO",
   "layout": "5pt_full",
   "sources": {
    "humanml": "HumanML3D",
    "lingo": "LINGO"
   }
  }
 },
 "methods": [
  {
   "key": "gt",
   "display_name": "Ground Truth",
   "color": "#E6A23C"
  },
  {
   "key": "ours",
   "display_name": "Ours",
   "color": "#16A085"
  }
 ],
 "samples": [
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/M011038_1",
   "slug": "humanml_M011038_1",
   "layout": "5pt_full",
   "caption": "A person is performing a crossover move while dribbling a basketball.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "A person is performing a crossover move while dribbling a basketball.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 ball",
     "color": "#16A085",
     "text": "A person dribbles the ball with his right hand before shooting with both.",
     "note": ""
    }
   ],
   "frames": 120,
   "duration_seconds": 4.0,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_M011038_1/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_M011038_1/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/M004630_2",
   "slug": "humanml_M004630_2",
   "layout": "5pt_full",
   "caption": "A person walks forward and leans over a table with their right hand on it.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "A person walks forward and leans over a table with their right hand on it.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 table",
     "color": "#16A085",
     "text": "A person moves forward, places their right hand on a table, and uses their left hand to pick up an object.",
     "note": ""
    }
   ],
   "frames": 280,
   "duration_seconds": 9.33,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_M004630_2/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_M004630_2/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "LINGO/17555",
   "slug": "lingo_17555",
   "layout": "5pt_full",
   "caption": "play guitar with both hands",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "play guitar with both hands",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 chair",
     "color": "#16A085",
     "text": "A person plays guitar with both hands while sitting.",
     "note": ""
    }
   ],
   "frames": 148,
   "duration_seconds": 4.93,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/lingo_17555/comparison.mp4",
   "poster": "static/videos/ood_object/lingo_17555/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/M001632_1",
   "slug": "humanml_M001632_1",
   "layout": "5pt_full",
   "caption": "He takes a seat in an armchair.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "He takes a seat in an armchair.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 chair",
     "color": "#16A085",
     "text": "A person settles down in the armchair.",
     "note": ""
    }
   ],
   "frames": 300,
   "duration_seconds": 10.0,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_M001632_1/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_M001632_1/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/000095_1",
   "slug": "humanml_000095_1",
   "layout": "5pt_full",
   "caption": "They sit down and scan their surroundings.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "They sit down and scan their surroundings.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 chair",
     "color": "#16A085",
     "text": "A person is seated for a right knee reflex test while seated.",
     "note": ""
    }
   ],
   "frames": 300,
   "duration_seconds": 10.0,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_000095_1/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_000095_1/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/M003761_1",
   "slug": "humanml_M003761_1",
   "layout": "5pt_full",
   "caption": "A person sits on a stool with their right leg elevated, then adjusts their position to switch legs.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "A person sits on a stool with their right leg elevated, then adjusts their position to switch legs.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 chair",
     "color": "#16A085",
     "text": "A person adjusts their position on the stool with their right leg up, before switching legs.",
     "note": ""
    }
   ],
   "frames": 300,
   "duration_seconds": 10.0,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_M003761_1/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_M003761_1/poster.jpg"
  },
  {
   "dataset": "ood",
   "dataset_name": "HumanML3D + LINGO",
   "sample_id": "MotionUnion/humanml/007198_0",
   "slug": "humanml_007198_0",
   "layout": "5pt_full",
   "caption": "The person sits down, resting their arms on the armrests, and then gets back up.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT (no object annotation)",
     "color": "#E6A23C",
     "text": "The person sits down, resting their arms on the armrests, and then gets back up.",
     "note": ""
    },
    {
     "key": "ours",
     "label": "Ours \u00b7 chair",
     "color": "#16A085",
     "text": "A person sits down, places arms on the armrests, and then rises.",
     "note": ""
    }
   ],
   "frames": 160,
   "duration_seconds": 5.33,
   "roles": [
    "gt",
    "ours"
   ],
   "comparison_video": "static/videos/ood_object/humanml_007198_0/comparison.mp4",
   "poster": "static/videos/ood_object/humanml_007198_0/poster.jpg"
  }
 ]
};
