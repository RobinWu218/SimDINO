import data from '../../template.yaml';
import 'katex/dist/katex.min.css';
import UIkit from 'uikit';
import Icons from 'uikit/dist/js/uikit-icons';

const currentTheme = data.theme || 'default';

const loadTheme = async (theme) => {
  try {
    switch (theme) {
      case 'dark':
        await import('@/scss/dark-theme.scss');
        break;
      case 'classic':
        await import('@/scss/classic-theme.scss');
        break;
      default:
        await import('@/scss/theme.scss');
    }
  } catch (err) {
    console.error('Failed to load theme:', err);
  }
};

// Initialize theme and UIkit
loadTheme(currentTheme);
UIkit.use(Icons);

export const toggleTheme = () => {
  const themes = ['light', 'dark', 'classic'];
  const currentTheme = localStorage.getItem('theme') || 'light';
  const nextTheme = themes[(themes.indexOf(currentTheme) + 1) % themes.length];
  loadTheme(nextTheme);
  return nextTheme;
};
