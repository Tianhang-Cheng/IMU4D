window.SWAP_MANIFEST = 
{
 "schema_version": 1,
 "title": "Text-conditioned object swaps",
 "protocol": {
  "layout": "5pt_full"
 },
 "datasets": {
  "omomo": {
   "display_name": "OMOMO",
   "layout": "5pt_full"
  },
  "hiphi": {
   "display_name": "HiPHI",
   "layout": "5pt_full"
  }
 },
 "methods": [
  {
   "key": "gt",
   "display_name": "Ground Truth",
   "color": "#E6A23C"
  },
  {
   "key": "orig",
   "display_name": "Ours",
   "color": "#16A085"
  }
 ],
 "samples": [
  {
   "dataset": "omomo",
   "dataset_name": "OMOMO",
   "sample_id": "OMOMO/sub17_suitcase_000",
   "slug": "OMOMO_sub17_suitcase_000-0062fd6a",
   "layout": "5pt_full",
   "caption": "A person pushes the suitcase and sets it back down.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person pushes the suitcase and sets it back down.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person pushes the suitcase and sets it back down.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Suitcase \u2192 trash can",
     "color": "#2563EB",
     "text": "A person pushes the trashcan and sets it back down.",
     "note": ""
    },
    {
     "key": "side_table",
     "label": "Suitcase \u2192 side table",
     "color": "#2563EB",
     "text": "A person pushes the smalltable and sets it back down.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Suitcase \u2192 box",
     "color": "#2563EB",
     "text": "A person pushes the largebox and sets it back down.",
     "note": ""
    }
   ],
   "frames": 144,
   "duration_seconds": 4.8,
   "roles": [
    "gt",
    "orig",
    "trash_can",
    "side_table",
    "box"
   ],
   "comparison_video": "static/videos/scene_swap/omomo/OMOMO_sub17_suitcase_000-0062fd6a/comparison.mp4",
   "poster": "static/videos/scene_swap/omomo/OMOMO_sub17_suitcase_000-0062fd6a/poster.jpg"
  },
  {
   "dataset": "omomo",
   "dataset_name": "OMOMO",
   "sample_id": "OMOMO/sub16_largebox_044",
   "slug": "OMOMO_sub16_largebox_044-2388868a",
   "layout": "5pt_full",
   "caption": "A person lifts the largebox, moves it, and puts it down.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person lifts the largebox, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person lifts the smalltable, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Swap \u2192 trash can",
     "color": "#2563EB",
     "text": "A person lifts the trashcan, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "suitcase",
     "label": "Swap \u2192 suitcase",
     "color": "#2563EB",
     "text": "A person lifts the suitcase, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Swap \u2192 box",
     "color": "#2563EB",
     "text": "A person lifts the largebox, moves it, and puts it down.",
     "note": ""
    }
   ],
   "frames": 136,
   "duration_seconds": 4.533333333333333,
   "roles": [
    "gt",
    "orig",
    "trash_can",
    "suitcase",
    "box"
   ],
   "comparison_video": "static/videos/scene_swap/omomo/OMOMO_sub16_largebox_044-2388868a/comparison.mp4",
   "poster": "static/videos/scene_swap/omomo/OMOMO_sub16_largebox_044-2388868a/poster.jpg"
  },
  {
   "dataset": "omomo",
   "dataset_name": "OMOMO",
   "sample_id": "OMOMO/sub17_smalltable_038",
   "slug": "OMOMO_sub17_smalltable_038-c6a09f2d",
   "layout": "5pt_full",
   "caption": "A person pushes the smalltable, releases the hands, then drags the smalltable and sets it back down.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person pushes the smalltable, releases the hands, then drags the smalltable and sets it back down.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person pushes the plasticbox, releases the hands, then pulls the plasticbox and sets it back down.",
     "note": ""
    },
    {
     "key": "side_table",
     "label": "Swap \u2192 side table",
     "color": "#2563EB",
     "text": "A person pushes the smalltable, releases the hands, then pulls the smalltable and sets it back down.",
     "note": ""
    },
    {
     "key": "suitcase",
     "label": "Swap \u2192 suitcase",
     "color": "#2563EB",
     "text": "A person pushes the suitcase, releases the hands, then pulls the suitcase and sets it back down.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Swap \u2192 trash can",
     "color": "#2563EB",
     "text": "A person pushes the trashcan, releases the hands, then pulls the trashcan and sets it back down.",
     "note": ""
    }
   ],
   "frames": 252,
   "duration_seconds": 8.4,
   "roles": [
    "gt",
    "orig",
    "side_table",
    "suitcase",
    "trash_can"
   ],
   "comparison_video": "static/videos/scene_swap/omomo/OMOMO_sub17_smalltable_038-c6a09f2d/comparison.mp4",
   "poster": "static/videos/scene_swap/omomo/OMOMO_sub17_smalltable_038-c6a09f2d/poster.jpg"
  },
  {
   "dataset": "omomo",
   "dataset_name": "OMOMO",
   "sample_id": "OMOMO/sub16_trashcan_016",
   "slug": "OMOMO_sub16_trashcan_016-5282628e",
   "layout": "5pt_full",
   "caption": "A person kicks the trashcan, lifts it, moves it, and puts it down.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person kicks the trashcan, lifts it, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person kicks the trashcan, lifts it, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Trashcan \u2192 box",
     "color": "#2563EB",
     "text": "A person kicks the largebox, lifts it, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "suitcase",
     "label": "Trashcan \u2192 suitcase",
     "color": "#2563EB",
     "text": "A person kicks the suitcase, lifts it, moves it, and puts it down.",
     "note": ""
    },
    {
     "key": "chair",
     "label": "Trashcan \u2192 chair",
     "color": "#2563EB",
     "text": "A person kicks the woodchair, lifts it, moves it, and puts it down.",
     "note": ""
    }
   ],
   "frames": 192,
   "duration_seconds": 6.4,
   "roles": [
    "gt",
    "orig",
    "box",
    "suitcase",
    "chair"
   ],
   "comparison_video": "static/videos/scene_swap/omomo/OMOMO_sub16_trashcan_016-5282628e/comparison.mp4",
   "poster": "static/videos/scene_swap/omomo/OMOMO_sub16_trashcan_016-5282628e/poster.jpg"
  },
  {
   "dataset": "hiphi",
   "dataset_name": "HiPHI",
   "sample_id": "HiPHI/Bringing-lug_0391",
   "slug": "HiPHI_Bringing-lug_0391-2943247d",
   "layout": "5pt_full",
   "caption": "A person lugs the chair low and moves forward and back in a fast shuttle with long, forceful strides, repeatedly.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person lugs the chair low and moves forward and back in a fast shuttle with long, forceful strides, repeatedly.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person humps the chair forward with long, driving steps and a pronounced full-body lean, repeatedly.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Chair \u2192 box",
     "color": "#2563EB",
     "text": "A person humps the box forward with long, driving steps and a pronounced full-body lean, repeatedly.",
     "note": ""
    },
    {
     "key": "table",
     "label": "Chair \u2192 table",
     "color": "#2563EB",
     "text": "A person humps the table forward with long, driving steps and a pronounced full-body lean, repeatedly.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Chair \u2192 trash can",
     "color": "#2563EB",
     "text": "A person humps the trash bin forward with long, driving steps and a pronounced full-body lean, repeatedly.",
     "note": ""
    }
   ],
   "frames": 480,
   "duration_seconds": 16.0,
   "roles": [
    "gt",
    "orig",
    "box",
    "table",
    "trash_can"
   ],
   "comparison_video": "static/videos/scene_swap/hiphi/HiPHI_Bringing-lug_0391-2943247d/comparison.mp4",
   "poster": "static/videos/scene_swap/hiphi/HiPHI_Bringing-lug_0391-2943247d/poster.jpg"
  },
  {
   "dataset": "hiphi",
   "dataset_name": "HiPHI",
   "sample_id": "HiPHI/Bringing-carry_0099",
   "slug": "HiPHI_Bringing-carry_0099-137c7f3b",
   "layout": "5pt_full",
   "caption": "A person walks to the box, lifts it from 15 cm off the floor, carries it forward with steady steps while keeping it in front of you, and moves it to 45 cm off the floor, repeatedly.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person walks to the box, lifts it from 15 cm off the floor, carries it forward with steady steps while keeping it in front of you, and moves it to 45 cm off the floor, repeatedly.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person takes the chair with them forward at a fast walk with long strides and strong arm control, repeatedly.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Swap \u2192 box",
     "color": "#2563EB",
     "text": "A person takes the box with them forward at a fast walk with long strides and strong arm control, repeatedly.",
     "note": ""
    },
    {
     "key": "table",
     "label": "Swap \u2192 table",
     "color": "#2563EB",
     "text": "A person takes the table with them forward at a fast walk with long strides and strong arm control, repeatedly.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Swap \u2192 trash can",
     "color": "#2563EB",
     "text": "A person takes the trash bin with them forward at a fast walk with long strides and strong arm control, repeatedly.",
     "note": ""
    }
   ],
   "frames": 480,
   "duration_seconds": 16.0,
   "roles": [
    "gt",
    "orig",
    "box",
    "table",
    "trash_can"
   ],
   "comparison_video": "static/videos/scene_swap/hiphi/HiPHI_Bringing-carry_0099-137c7f3b/comparison.mp4",
   "poster": "static/videos/scene_swap/hiphi/HiPHI_Bringing-carry_0099-137c7f3b/poster.jpg"
  },
  {
   "dataset": "hiphi",
   "dataset_name": "HiPHI",
   "sample_id": "HiPHI/Bringing-carry_0105",
   "slug": "HiPHI_Bringing-carry_0105-36faec98",
   "layout": "5pt_full",
   "caption": "A person walks to the box, lifts it from the floor, walks in a smooth curved path while keeping the box in front of them, and sets it down at about 45 cm off the floor, repeatedly.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person walks to the box, lifts it from the floor, walks in a smooth curved path while keeping the box in front of them, and sets it down at about 45 cm off the floor, repeatedly.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person walks up to the box on the ground, lifts it with both hands, carries it in small loops with steady steps, sets it back on the ground, and returns upright, repeatedly.",
     "note": ""
    },
    {
     "key": "chair",
     "label": "Box \u2192 chair",
     "color": "#2563EB",
     "text": "A person walks up to the chair on the ground, lifts it with both hands, carries it in small loops with steady steps, sets it back on the ground, and returns upright, repeatedly.",
     "note": ""
    },
    {
     "key": "table",
     "label": "Box \u2192 table",
     "color": "#2563EB",
     "text": "A person walks up to the table on the ground, lifts it with both hands, carries it in small loops with steady steps, sets it back on the ground, and returns upright, repeatedly.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Box \u2192 trash can",
     "color": "#2563EB",
     "text": "A person walks up to the trash bin on the ground, lifts it with both hands, carries it in small loops with steady steps, sets it back on the ground, and returns upright, repeatedly.",
     "note": ""
    }
   ],
   "frames": 480,
   "duration_seconds": 16.0,
   "roles": [
    "gt",
    "orig",
    "chair",
    "table",
    "trash_can"
   ],
   "comparison_video": "static/videos/scene_swap/hiphi/HiPHI_Bringing-carry_0105-36faec98/comparison.mp4",
   "poster": "static/videos/scene_swap/hiphi/HiPHI_Bringing-carry_0105-36faec98/poster.jpg"
  },
  {
   "dataset": "hiphi",
   "dataset_name": "HiPHI",
   "sample_id": "HiPHI/Cause_motion-lift_0013",
   "slug": "HiPHI_Cause_motion-lift_0013-7e16776d",
   "layout": "5pt_full",
   "caption": "A person lifts the chair from the floor with a deep squat and stands tall before setting it down, repeatedly.",
   "text_results": [
    {
     "key": "gt",
     "label": "GT",
     "color": "#E6A23C",
     "text": "A person lifts the chair from the floor with a deep squat and stands tall before setting it down, repeatedly.",
     "note": ""
    },
    {
     "key": "orig",
     "label": "Ours",
     "color": "#16A085",
     "text": "A person lifts the chair up to chest height and lowers it back to the floor, repeatedly.",
     "note": ""
    },
    {
     "key": "box",
     "label": "Swap \u2192 box",
     "color": "#2563EB",
     "text": "A person lifts the box up to chest height and lowers it back to the floor, repeatedly.",
     "note": ""
    },
    {
     "key": "table",
     "label": "Swap \u2192 table",
     "color": "#2563EB",
     "text": "A person lifts the table up to chest height and lowers it back to the floor, repeatedly.",
     "note": ""
    },
    {
     "key": "trash_can",
     "label": "Swap \u2192 trash can",
     "color": "#2563EB",
     "text": "A person lifts the trash bin up to chest height and lowers it back to the floor, repeatedly.",
     "note": ""
    }
   ],
   "frames": 480,
   "duration_seconds": 16.0,
   "roles": [
    "gt",
    "orig",
    "box",
    "table",
    "trash_can"
   ],
   "comparison_video": "static/videos/scene_swap/hiphi/HiPHI_Cause_motion-lift_0013-7e16776d/comparison.mp4",
   "poster": "static/videos/scene_swap/hiphi/HiPHI_Cause_motion-lift_0013-7e16776d/poster.jpg"
  }
 ]
}
