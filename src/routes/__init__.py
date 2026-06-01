from .pages import pages_bp
from .search import search_bp
from .browse import browse_bp
from .stats import stats_bp
from .manage import manage_bp
from .debug import debug_bp
from .search_cascaded import search_cascaded_bp  # ← thêm dòng này

def register_routes(app):
    app.register_blueprint(pages_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(browse_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(manage_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(search_cascaded_bp)   # ← thêm dòng này