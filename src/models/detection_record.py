from src.extensions import db
from .user import utc_now


class DetectionRecord(db.Model):
    __tablename__ = "detection_records"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    detection_type = db.Column(db.String(10), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    original_path = db.Column(db.String(255), nullable=False)
    result_path = db.Column(db.String(255), nullable=False)
    preview_path = db.Column(db.String(255))
    total_objects = db.Column(db.Integer, nullable=False)
    class_counts = db.Column(db.JSON, nullable=False)
    detections = db.Column(db.JSON, nullable=False, default=list)
    processing_time = db.Column(db.Float, nullable=False)
    inference_time = db.Column(db.Float, nullable=False)
    confidence_threshold = db.Column(db.Float, nullable=False)
    mean_confidence = db.Column(db.Float)
    media_info = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, nullable=False, default=utc_now, index=True)

    @property
    def is_video(self):
        return self.detection_type == "video"
