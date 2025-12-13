// Sidebar Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mainContent = document.getElementById('mainContent');
    
    // Load saved sidebar state from localStorage
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (sidebarCollapsed) {
        sidebar.classList.add('sidebar-collapsed');
    }
    
    // Toggle sidebar on button click
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('sidebar-collapsed');
            
            // Save state to localStorage
            const isCollapsed = sidebar.classList.contains('sidebar-collapsed');
            localStorage.setItem('sidebarCollapsed', isCollapsed);
        });
    }
    
    // Mobile menu toggle (for screens < 1024px)
    const createMobileMenuButton = () => {
        if (window.innerWidth <= 1024 && !document.getElementById('mobileMenuBtn')) {
            const mobileBtn = document.createElement('button');
            mobileBtn.id = 'mobileMenuBtn';
            mobileBtn.className = 'mobile-menu-btn';
            mobileBtn.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
            `;
            mobileBtn.style.cssText = `
                position: fixed;
                top: 1rem;
                left: 1rem;
                z-index: 101;
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #1e40af, #3b82f6);
                border: none;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                color: white;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            `;
            
            document.body.appendChild(mobileBtn);
            
            mobileBtn.addEventListener('click', () => {
                sidebar.classList.toggle('mobile-open');
            });
            
            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 1024 && 
                    sidebar.classList.contains('mobile-open') &&
                    !sidebar.contains(e.target) && 
                    !mobileBtn.contains(e.target)) {
                    sidebar.classList.remove('mobile-open');
                }
            });
        }
    };
    
    // Check on load and resize
    createMobileMenuButton();
    window.addEventListener('resize', createMobileMenuButton);
});
