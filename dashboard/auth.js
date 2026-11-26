/* Who is logged in, for every page (docs/access.md). Served by the UTM server: asks GET api/me; no session sends the page to
   login.html. Opened without the server (file:// or a plain static server, from data/*.json): there is no login, the page is
   the admin's offline view and says so in a banner. With the server's --no-auth everyone is the admin and a banner says that.

     const me = await Auth.init();     // { role, admin, offline, noAuth }
     me.admin || me.offline            // may see every operator
*/
const Auth = (() => {
  let me = { role: null, admin: true, offline: true, noAuth: false };
  const css = `
.authbar { white-space: nowrap; display: inline-flex; align-items: center; gap: 8px; font: 400 13px/1.2 var(--font, system-ui, sans-serif); color: var(--muted, #6b6b6b); }
.authbar .authname { font: 500 13px/1.2 var(--font, system-ui, sans-serif); color: var(--text, #111); display: inline; }
.authbar button { font: inherit; border: 0; border-radius: 6px; padding: 5px 10px; background: var(--raised, #ecebe8); color: var(--text, #111); cursor: pointer; }
.authbar button:hover { background: var(--border, #ddd); }
.authnote { padding: 7px 16px; font: 400 13px/1.35 var(--font, system-ui, sans-serif); text-align: center; background: var(--warn-soft, #fbf1e3); color: var(--warn, #92400e); }
.authnote.danger { background: var(--danger-soft, #fbe9e9); color: var(--danger, #c4242a); font-weight: 500; }`;

  function banner(text, danger) {
    const d = document.createElement('div'); d.className = 'authnote' + (danger ? ' danger' : ''); d.textContent = text; d.setAttribute('role', 'status');
    document.body.insertBefore(d, document.body.firstChild);
  }

  function bar() {
    const b = document.createElement('span'); b.className = 'authbar';
    const who = document.createElement('span'); who.append('Logged in as ');
    const name = document.createElement('span'); name.className = 'authname'; name.textContent = me.role + (me.admin ? ' (admin)' : ''); who.append(name);
    const out = document.createElement('button'); out.type = 'button'; out.textContent = 'Log out';
    out.onclick = async () => { try { await fetch('api/logout', { method: 'POST', body: '{}' }); } catch { } location.href = 'login.html'; };
    b.append(who, out);
    const host = document.querySelector('header') || document.querySelector('.brand');
    if (host) { if (host.tagName === 'HEADER') host.append(b); else { b.style.margin = '6px 0'; host.after(b); } }
    else document.body.insertBefore(b, document.body.firstChild);
  }

  function toLogin() {
    const next = location.pathname.split('/').pop() + location.search + location.hash;
    location.replace('login.html?next=' + encodeURIComponent(next));
    return new Promise(() => { });                       // the page is leaving: never carry on
  }

  async function init({ login = false } = {}) {
    const st = document.createElement('style'); st.textContent = css; document.head.append(st);
    let r = null;
    try { r = await fetch('api/me', { cache: 'no-store' }); } catch { r = null; }
    const json = r && (r.headers.get('content-type') || '').includes('json');
    if (!json) {                                          // no UTM server: the offline view (data/*.json), no login
      me = { role: 'admin', admin: true, offline: true, noAuth: false };
      if (!login) banner('Replay mode: recorded flights, opened without the UTM server, so there is no login. You see every operator, as the admin does.');
      return me;
    }
    if (r.status === 401) { if (login) return (me = { role: null, admin: false, offline: false, noAuth: false }); return toLogin(); }
    const j = await r.json();
    me = { role: j.role, admin: !!j.admin, offline: false, noAuth: !!j.no_auth };
    if (login) return me;
    if (me.noAuth) banner('WARNING: the server runs with --no-auth: there is no login, everyone is the admin and sees every operator. Demo use only.', true);
    else bar();
    const real = window.fetch;                            // a session that ended (8 h, or a restart): back to the login page
    window.fetch = async (...a) => {
      const res = await real(...a);
      if (res.status === 401 && !me.noAuth && String(a[0]).startsWith('api/') && !String(a[0]).startsWith('api/login')) toLogin();
      return res;
    };
    return me;
  }
  return { init, get me() { return me; } };
})();
