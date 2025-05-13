import numpy as np
# Import necessary classes and functions from other modules
from detection_parser import ParsedObject, filter_and_parse_detections
from spatial_analyzer import get_main_walk_area, get_object_walk_area_relation, get_relative_position, get_distance_category, calculate_iou

# Navigation Instructions (constants for clarity)
NAV_PROCEED_STRAIGHT = "PROCEED_STRAIGHT"
NAV_TURN_LEFT = "TURN_LEFT"
NAV_TURN_RIGHT = "TURN_RIGHT"
NAV_VEER_LEFT = "VEER_LEFT"
NAV_VEER_RIGHT = "VEER_RIGHT"
NAV_SLOW_DOWN = "SLOW_DOWN"
NAV_STOP = "STOP"
NAV_CAUTION = "CAUTION"
NAV_SEARCHING_PATH = "SEARCHING_PATH"

# Warning Levels (constants for clarity)
WARN_NONE = "NONE"
WARN_LOW = "LOW"
WARN_MEDIUM = "MEDIUM"
WARN_HIGH = "HIGH"
WARN_CRITICAL = "CRITICAL"

class NavigationDecision:
    """Represents the final navigation advice for the user."""
    def __init__(self, instruction, warning_level, message, details=None):
        self.instruction = instruction
        self.warning_level = warning_level
        self.message = message
        self.details = details if details is not None else {}

    def __repr__(self):
        return f"NavigationDecision(instruction=\"{self.instruction}\", warning=\"{self.warning_level}\", message=\"{self.message}\")"

def generate_navigation_decision(parsed_objects, image_width, image_height):
    """Generates a navigation decision based on parsed objects and their spatial relationships.

    Args:
        parsed_objects (list): A list of ParsedObject instances from detection_parser.
        image_width (int): The width of the image frame.
        image_height (int): The height of the image frame.

    Returns:
        NavigationDecision: The calculated navigation decision.
    """
    main_walk_area = get_main_walk_area(parsed_objects)

    # Minimum acceptable walk area size (e.g., 5% of total image area)
    min_walk_area_size = image_width * image_height * 0.05

    if main_walk_area is None or main_walk_area.area < min_walk_area_size:
        return NavigationDecision(NAV_STOP, WARN_CRITICAL, "没有检测到足够的可行走区域，请停止！", 
                                {"reason": "No viable walk area or walk area too small", "walk_area_found": main_walk_area is not None, "walk_area_area": main_walk_area.area if main_walk_area else 0})

    obstacles_and_humans = [obj for obj in parsed_objects if obj.class_name in ["obstacle", "human"]]
    
    threats = []
    for obj in obstacles_and_humans:
        # Ensure obj.mask is valid for spatial analysis
        if obj.mask is None or obj.area == 0:
            # print(f"Skipping object {obj.class_name} due to invalid mask or zero area for threat analysis.")
            continue

        relation = get_object_walk_area_relation(obj, main_walk_area.mask)
        relative_pos = get_relative_position(obj.centroid[0], image_width)
        
        # Determine if the object is on the primary path
        # Simplified: on path if inside/edge of walk_area AND in the center horizontal zone
        is_on_primary_path = (relation in ["INSIDE", "EDGE"]) and (relative_pos == "CENTER")
        distance_cat = get_distance_category(obj, image_height, is_on_primary_path)
        
        threat_score = 0
        # Score based on relation to walk area
        if relation == "INSIDE":
            threat_score += 50
        elif relation == "EDGE":
            threat_score += 30
        elif relation == "NEAR_OUTSIDE": # Close enough to be a concern
            threat_score += 10
        
        # Score based on distance
        if distance_cat == "NEAR":
            threat_score += 40
        elif distance_cat == "MID":
            threat_score += 20
        
        # Score based on horizontal position relative to user
        if relative_pos == "CENTER":
            threat_score += 25 # Increased score for objects directly in front
        elif is_on_primary_path: # Object is on path but not perfectly centered (e.g. slightly to left/right of center path)
             threat_score += 15
        
        if obj.class_name == "human":
            threat_score += 10 # Humans are more dynamic, slightly higher base threat

        if threat_score > 15: # Minimum score to be considered a threat
            threats.append({
                "obj": obj,
                "relation": relation,
                "relative_pos": relative_pos,
                "distance": distance_cat,
                "score": threat_score,
                "is_on_primary_path": is_on_primary_path
            })

    if not threats:
        # Basic check for walk area continuity and direction if no immediate threats
        wa_centroid_x, wa_centroid_y = main_walk_area.centroid
        
        # If walk area is substantially in the lower half and wide enough centrally
        if wa_centroid_y > image_height / 2 and main_walk_area.area > image_width * image_height * 0.20:
            center_x_start = image_width // 3
            center_x_end = 2 * image_width // 3
            walk_area_center_strip_mask = main_walk_area.mask[:, center_x_start:center_x_end]
            if np.sum(walk_area_center_strip_mask) / (walk_area_center_strip_mask.size + 1e-6) > 0.4: # 40% of central strip is walkable
                return NavigationDecision(NAV_PROCEED_STRAIGHT, WARN_NONE, "前方安全，请直行。")
        
        return NavigationDecision(NAV_CAUTION, WARN_LOW, "前方暂无明显障碍，请谨慎前行，注意观察。", {"reason": "Clear of major threats, general caution advised"})

    threats.sort(key=lambda t: t["score"], reverse=True)
    primary_threat = threats[0]
    pt_obj_name = primary_threat["obj"].class_name
    pt_dist = primary_threat["distance"]
    pt_pos = primary_threat["relative_pos"]
    pt_relation = primary_threat["relation"]

    # Decision logic based on the primary (highest score) threat
    if pt_dist == "NEAR" and pt_pos == "CENTER" and pt_relation == "INSIDE":
        return NavigationDecision(NAV_STOP, WARN_CRITICAL, f"停止！正前方近距离有{pt_obj_name}！", {"threat_details": primary_threat})

    if pt_dist == "NEAR" and pt_relation == "INSIDE":
        if pt_pos == "LEFT":
            # Check if right side is clear enough to veer right
            # This requires checking for other threats on the right or if walk_area allows veering right
            return NavigationDecision(NAV_VEER_RIGHT, WARN_HIGH, f"注意！左前方近处有{pt_obj_name}，建议向右微调。", {"threat_details": primary_threat})
        elif pt_pos == "RIGHT":
            return NavigationDecision(NAV_VEER_LEFT, WARN_HIGH, f"注意！右前方近处有{pt_obj_name}，建议向左微调。", {"threat_details": primary_threat})
        else: # Center, but not critical enough for STOP (e.g., slightly further than immediate stop)
             return NavigationDecision(NAV_SLOW_DOWN, WARN_HIGH, f"注意！正前方近处有{pt_obj_name}，请减速！", {"threat_details": primary_threat})

    if pt_dist == "NEAR" and (pt_relation == "EDGE" or pt_relation == "NEAR_OUTSIDE") :
         return NavigationDecision(NAV_CAUTION, WARN_MEDIUM, f"注意！{pt_pos}侧边缘近处有{pt_obj_name}。", {"threat_details": primary_threat})

    if pt_dist == "MID" and pt_relation == "INSIDE":
        return NavigationDecision(NAV_SLOW_DOWN, WARN_MEDIUM, f"注意前方{pt_pos}有{pt_obj_name}，请减速慢行。", {"threat_details": primary_threat})

    if pt_dist == "MID" and (pt_relation == "EDGE" or pt_relation == "NEAR_OUTSIDE"):
        return NavigationDecision(NAV_CAUTION, WARN_LOW, f"注意{pt_pos}侧边缘有{pt_obj_name}。", {"threat_details": primary_threat})
    
    # Default for other significant threats not caught by specific rules above
    if primary_threat["score"] > 40: # General threshold for a notable threat
        return NavigationDecision(NAV_CAUTION, WARN_MEDIUM, f"请注意环境，前方{pt_pos}区域检测到{pt_obj_name} ({pt_dist})。", {"threat_details": primary_threat})

    # Fallback if no specific high-priority instruction was generated but threats exist
    return NavigationDecision(NAV_PROCEED_STRAIGHT, WARN_LOW, "路况基本清晰，但请注意观察周围环境。", {"reason": "Low level threats detected, proceed with general caution", "threat_details": primary_threat})


if __name__ == "__main__":
    # This example usage will now use the actual imported functions
    _img_width = 640
    _img_height = 480

    # --- Define Dummy Masks (ensure they are 0 or 1 for area calculation) ---
    def create_mask(coords_list, shape): # Helper to create mask from list of [y1,y2,x1,x2]
        mask = np.zeros(shape, dtype=np.uint8)
        for y1,y2,x1,x2 in coords_list:
            mask[y1:y2, x1:x2] = 1
        return mask

    walk_area_full_mask = create_mask([[240, _img_height, 0, _img_width]], (_img_height, _img_width))
    obstacle_critical_mask = create_mask([[400, 460, 300, 340]], (_img_height, _img_width))
    human_veer_left_mask = create_mask([[380, 450, 400, 450]], (_img_height, _img_width))
    obstacle_slow_mask = create_mask([[300, 350, 300, 340]], (_img_height, _img_width))
    small_wa_mask = create_mask([[400,410,300,310]], (_img_height, _img_width)) # Very small walk area
    obstacle_on_edge_mask = create_mask([[300,350, 580,620]], (_img_height, _img_width)) # Obstacle on right edge

    # --- Test Scenarios ---
    test_scenarios = {
        "Scenario 1: Critical Stop": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [320, 360, 640, 240], "mask": walk_area_full_mask},
            {"class_id": 1, "class_name": "obstacle", "confidence": 0.8, "bbox": [320, 430, 40, 60], "mask": obstacle_critical_mask}
        ],
        "Scenario 2: Veer Left (Human on Right)": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [320, 360, 640, 240], "mask": walk_area_full_mask},
            {"class_id": 2, "class_name": "human", "confidence": 0.85, "bbox": [425, 415, 50, 70], "mask": human_veer_left_mask}
        ],
        "Scenario 3: Slow Down (Obstacle Mid-Center)": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [320, 360, 640, 240], "mask": walk_area_full_mask},
            {"class_id": 1, "class_name": "obstacle", "confidence": 0.7, "bbox": [320, 325, 40, 50], "mask": obstacle_slow_mask}
        ],
        "Scenario 4: Clear Path": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [320, 360, 640, 240], "mask": walk_area_full_mask}
        ],
        "Scenario 5: No Walk Area": [
            {"class_id": 1, "class_name": "obstacle", "confidence": 0.8, "bbox": [320, 430, 40, 60], "mask": obstacle_critical_mask}
        ],
        "Scenario 6: Walk Area Too Small": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [305, 405, 10, 10], "mask": small_wa_mask}
        ],
        "Scenario 7: Obstacle on Edge": [
            {"class_id": 0, "class_name": "walk_area", "confidence": 0.9, "bbox": [320, 360, 640, 240], "mask": walk_area_full_mask},
            {"class_id": 1, "class_name": "obstacle", "confidence": 0.75, "bbox": [600, 325, 40, 50], "mask": obstacle_on_edge_mask}
        ]
    }

    for name, raw_detections_data in test_scenarios.items():
        print(f"\n--- {name} ---")
        # 1. Parse Detections
        parsed_objs = filter_and_parse_detections(raw_detections_data, _img_width, _img_height, confidence_threshold=0.5)
        # print("Parsed Objects:")
        # for p_obj in parsed_objs:
        #     print(f"  {p_obj}") # Relies on ParsedObject.__repr__ which includes area and centroid
        
        # 2. Generate Navigation Decision (which internally uses spatial_analyzer functions)
        decision = generate_navigation_decision(parsed_objs, _img_width, _img_height)
        print(decision)
        if decision.details:
            print(f"  Details: {decision.details}")

