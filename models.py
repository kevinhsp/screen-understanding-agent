"""
models.py - Core data structures and type definitions for screen understanding system
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ScreenType(Enum):
    """Screen type enumeration"""
    LOGIN = "login"
    DASHBOARD = "dashboard"
    FORM = "form"
    LIST = "list"
    DETAIL = "detail"
    SETTINGS = "settings"
    SEARCH = "search"
    CHECKOUT = "checkout"
    CHAT = "chat"
    MEDIA = "media"
    ERROR = "error"
    DIALOG = "dialog"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


class ElementRole(Enum):
    """UI element role enumeration"""
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    IMAGE = "image"
    TEXT = "text"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    MENU = "menu"
    TAB = "tab"
    ICON = "icon"
    CONTAINER = "container"
    LABEL = "label"
    HEADING = "heading"
    TABLE = "table"
    LIST_ITEM = "list_item"
    TOGGLE = "toggle"
    SLIDER = "slider"
    PROGRESS = "progress"


class ElementState(Enum):
    """UI element state enumeration"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    SELECTED = "selected"
    FOCUSED = "focused"
    HIDDEN = "hidden"
    LOADING = "loading"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"


class ActionType(Enum):
    """Possible action types"""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SWIPE = "swipe"
    DRAG = "drag"
    HOVER = "hover"
    LONG_PRESS = "long_press"
    DOUBLE_CLICK = "double_click"
    SELECT = "select"
    CLEAR = "clear"
    PASTE = "paste"
    NAVIGATE = "navigate"


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) format"""
        return (self.x, self.y, self.width, self.height)
    
    def area(self) -> int:
        """Calculate area of bounding box"""
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class OCRResult:
    """OCR recognition result"""
    text: str
    bbox: BoundingBox
    confidence: float
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "language": self.language
        }


@dataclass
class UIElement:
    """UI element representation"""
    id: str
    role: ElementRole
    text: str
    state: ElementState
    bbox: BoundingBox
    confidence: float = 0.0
    icon_type: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    clickable: bool = False
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "text": self.text,
            "state": self.state.value,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "icon_type": self.icon_type,
            "attributes": self.attributes,
            "clickable": self.clickable,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids
        }
    
    def is_interactive(self) -> bool:
        """Check if element is interactive"""
        interactive_roles = {
            ElementRole.BUTTON, ElementRole.INPUT, ElementRole.LINK,
            ElementRole.CHECKBOX, ElementRole.RADIO, ElementRole.DROPDOWN,
            ElementRole.TAB, ElementRole.TOGGLE, ElementRole.SLIDER
        }
        return self.role in interactive_roles and self.state != ElementState.DISABLED


@dataclass
class Affordance:
    """Executable action on screen"""
    action_type: ActionType
    target_element: Optional[str] = None  # element id
    description: str = ""
    priority: int = 5  # 0-10, higher is more important
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)  # list of prerequisite action descriptions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_element": self.target_element,
            "description": self.description,
            "priority": self.priority,
            "parameters": self.parameters,
            "prerequisites": self.prerequisites
        }


@dataclass
class ExtractedEntity:
    """Extracted key entity from screen"""
    entity_type: str  # date, price, email, phone, address, name, etc.
    value: str
    context: str = ""
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    source: str = "text"  # text, ocr, element, vlm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "context": self.context,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "source": self.source
        }


@dataclass
class ScreenUnderstanding:
    """Complete screen understanding result"""
    screen_type: ScreenType
    summary: str
    affordances: List[Affordance]
    entities: List[ExtractedEntity]
    elements: List[UIElement]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_type": self.screen_type.value,
            "summary": self.summary,
            "affordances": [a.to_dict() for a in self.affordances],
            "entities": [e.to_dict() for e in self.entities],
            "elements": [el.to_dict() for el in self.elements],
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms
        }
    
    def get_interactive_elements(self) -> List[UIElement]:
        """Get all interactive elements"""
        return [el for el in self.elements if el.is_interactive()]
    
    def get_elements_by_role(self, role: ElementRole) -> List[UIElement]:
        """Get elements by specific role"""
        return [el for el in self.elements if el.role == role]
    
    def get_high_priority_actions(self, threshold: int = 7) -> List[Affordance]:
        """Get high priority actions"""
        return [a for a in self.affordances if a.priority >= threshold]


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    # OCR settings
    ocr_language: str = "en"
    ocr_confidence_threshold: float = 0.5
    # Which OCR backend to use: auto | paddle | easyocr | trocr | tesseract | hybrid
    ocr_backend: str = "auto"
    
    # OmniParser settings
    element_detection_threshold: float = 0.3
    merge_overlapping_elements: bool = True
    iou_threshold: float = 0.5
    # OmniParser weights and source control
    omniparser_weights_path: Optional[str] = 'models/omniparser/icon_detect/model.pt'  # local path to YOLO weights
    omniparser_download_source: str = "hf"  # hf | github | none
    omniparser_allow_download: bool = True
    
    # VLM settings
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_max_tokens: int = 512
    vlm_temperature: float = 0.7
    use_multiple_vlms: bool = False
    # VLM prompt/selection knobs
    vlm_elements_max: int = 30
    vlm_clickable_only: bool = True
    vlm_min_label_len: int = 2
    vlm_prompt_example: bool = True
    vlm_json_tag: bool = True
    # Freeform control: whether to include OCR texts as extra context
    vlm_freeform_include_ocr: bool = False
    # Minimal DW-state output (summary+actions from VLM freeform, elements from OmniParser)
    dw_minimal_output: bool = False

    # Processing settings
    enable_entity_extraction: bool = True
    enable_affordance_detection: bool = True
    max_elements: int = 100
    debug_mode: bool = False
    save_intermediate_results: bool = False

    # Performance settings
    use_gpu: bool = True
    batch_size: int = 1
    num_workers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr_language": self.ocr_language,
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "ocr_backend": self.ocr_backend,
            "element_detection_threshold": self.element_detection_threshold,
            "merge_overlapping_elements": self.merge_overlapping_elements,
            "iou_threshold": self.iou_threshold,
            "omniparser_weights_path": self.omniparser_weights_path,
            "omniparser_download_source": self.omniparser_download_source,
            "omniparser_allow_download": self.omniparser_allow_download,
            "vlm_model_name": self.vlm_model_name,
            "vlm_max_tokens": self.vlm_max_tokens,
            "vlm_temperature": self.vlm_temperature,
            "use_multiple_vlms": self.use_multiple_vlms,
            "vlm_elements_max": self.vlm_elements_max,
            "vlm_clickable_only": self.vlm_clickable_only,
            "vlm_min_label_len": self.vlm_min_label_len,
            "vlm_prompt_example": self.vlm_prompt_example,
            "vlm_json_tag": self.vlm_json_tag,
            "vlm_freeform_include_ocr": self.vlm_freeform_include_ocr,
            "dw_minimal_output": self.dw_minimal_output,
            "enable_entity_extraction": self.enable_entity_extraction,
            "enable_affordance_detection": self.enable_affordance_detection,
            "max_elements": self.max_elements,
            "debug_mode": self.debug_mode,
            "save_intermediate_results": self.save_intermediate_results,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers
        }
