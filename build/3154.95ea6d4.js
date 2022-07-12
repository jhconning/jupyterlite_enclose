"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[3154,8393],{33154:(e,t,r)=>{r.r(t),r.d(t,{IServiceWorkerRegistrationWrapper:()=>c,JupyterLiteServer:()=>a,Router:()=>o,ServiceWorkerRegistrationWrapper:()=>u});var i=r(48231),s=r(71088),n=r(79334);class o{constructor(){this._routes=[]}get(e,t){this._add("GET",e,t)}put(e,t){this._add("PUT",e,t)}post(e,t){this._add("POST",e,t)}patch(e,t){this._add("PATCH",e,t)}delete(e,t){this._add("DELETE",e,t)}async route(e){const t=new URL(e.url),{method:r}=e,{pathname:i}=t;for(const s of this._routes){if(s.method!==r)continue;const n=i.match(s.pattern);if(!n)continue;const o=n.slice(1);let a;if("PATCH"===s.method||"PUT"===s.method||"POST"===s.method)try{a=JSON.parse(await e.text())}catch{a=void 0}return s.callback.call(null,{pathname:i,body:a,query:Object.fromEntries(t.searchParams)},...o)}throw new Error("Cannot route "+e.method+" "+e.url)}_add(e,t,r){"string"==typeof t&&(t=new RegExp(t)),this._routes.push({method:e,pattern:t,callback:r})}}class a extends s.Application{constructor(e){var t;super(e),this.name="JupyterLite Server",this.namespace=this.name,this.version="unknown",this._router=new o,this._serviceManager=new i.ServiceManager({standby:"never",serverSettings:{...i.ServerConnection.makeSettings(),WebSocket:n.WebSocket,fetch:null!==(t=this.fetch.bind(this))&&void 0!==t?t:void 0}})}get router(){return this._router}get serviceManager(){return this._serviceManager}async fetch(e,t){if(!(e instanceof Request))throw Error("Request info is not a Request");return this._router.route(e)}attachShell(e){}evtResize(e){}registerPluginModule(e){let t=e.default;Object.prototype.hasOwnProperty.call(e,"__esModule")||(t=e),Array.isArray(t)||(t=[t]),t.forEach((e=>{try{this.registerPlugin(e)}catch(e){console.error(e)}}))}registerPluginModules(e){e.forEach((e=>{this.registerPluginModule(e)}))}}const c=new(r(74547).Token)("@jupyterlite/server-extension:IServiceWorkerRegistrationWrapper");var h=r(58646),l=r(1005);class u{constructor(){this._registration=null,this._registrationChanged=new h.Signal(this),this.initialize()}get registrationChanged(){return this._registrationChanged}get enabled(){return null!==this._registration}async initialize(){if("serviceWorker"in navigator||(console.error("ServiceWorker registration failed: Service Workers not supported in this browser"),this.setRegistration(null)),navigator.serviceWorker.controller){const e=await navigator.serviceWorker.getRegistration(navigator.serviceWorker.controller.scriptURL);e&&this.setRegistration(e)}return await navigator.serviceWorker.register(l.URLExt.join(l.PageConfig.getBaseUrl(),"services.js")).then((e=>{this.setRegistration(e)}),(e=>{console.error(`ServiceWorker registration failed: ${e}`),this.setRegistration(null)}))}setRegistration(e){this._registration=e,this._registrationChanged.emit(this._registration)}}}}]);
//# sourceMappingURL=3154.95ea6d4.js.map