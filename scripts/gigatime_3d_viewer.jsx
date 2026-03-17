import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

const CHANNELS = [
  { name: "DAPI",       color: "#4169E1", desc: "Nuclear stain" },
  { name: "PD-1",       color: "#FF6B6B", desc: "Immune checkpoint" },
  { name: "CD14",       color: "#FFA726", desc: "Monocytes" },
  { name: "CD4",        color: "#66BB6A", desc: "Helper T-cells" },
  { name: "T-bet",      color: "#AB47BC", desc: "Th1 transcription" },
  { name: "CD34",       color: "#FFEE58", desc: "Stem/endothelial" },
  { name: "CD68",       color: "#FF1744", desc: "Macrophages" },
  { name: "CD16",       color: "#00E5FF", desc: "NK cells" },
  { name: "CD11c",      color: "#FF9100", desc: "Dendritic cells" },
  { name: "CD138",      color: "#E040FB", desc: "Plasma cells" },
  { name: "CD20",       color: "#00E676", desc: "B-cells" },
  { name: "CD3",        color: "#FF4081", desc: "Pan T-cells" },
  { name: "CD8",        color: "#00BCD4", desc: "Cytotoxic T-cells" },
  { name: "PD-L1",      color: "#FFD740", desc: "Immune ligand" },
  { name: "CK",         color: "#F48FB1", desc: "Cytokeratin/Epithelial" },
  { name: "Ki67",       color: "#B2FF59", desc: "Proliferation" },
  { name: "Tryptase",   color: "#FF6E40", desc: "Mast cells" },
  { name: "Actin-D",    color: "#18FFFF", desc: "Smooth muscle" },
  { name: "Caspase3-D", color: "#EA80FC", desc: "Apoptosis" },
  { name: "PHH3-B",     color: "#CCFF90", desc: "Mitosis" },
  { name: "Transgelin", color: "#FFD180", desc: "Stromal" },
];

function simulateChannels(imageData, width, height) {
  const r = new Float32Array(width*height), g = new Float32Array(width*height), b = new Float32Array(width*height);
  for (let i=0;i<width*height;i++) { r[i]=imageData[i*4]/255; g[i]=imageData[i*4+1]/255; b[i]=imageData[i*4+2]/255; }
  const hem=new Float32Array(width*height), eos=new Float32Array(width*height), tis=new Float32Array(width*height);
  for (let i=0;i<width*height;i++) {
    hem[i]=Math.max(0,Math.min(1,(1-r[i])*.6+(1-g[i])*.3+b[i]*.3));
    eos[i]=Math.max(0,Math.min(1,r[i]*.5+(1-g[i])*.3+(1-b[i])*.2));
    tis[i]=(r[i]+g[i]+b[i])/3<0.85?1:0;
  }
  const fns=[
    i=>hem[i]>.55?Math.pow(hem[i],1.5)*tis[i]:0, i=>hem[i]>.4&&eos[i]<.5?Math.random()*.3*hem[i]*tis[i]:0,
    i=>eos[i]>.45?Math.pow(eos[i],2)*.4*tis[i]*(Math.random()>.6?1:0):0, i=>hem[i]>.5?Math.random()*.35*tis[i]:0,
    i=>hem[i]>.6&&Math.random()>.7?.5*tis[i]:0, i=>eos[i]>.3&&hem[i]<.4?Math.random()*.25*tis[i]:0,
    i=>eos[i]>.5?Math.pow(eos[i],1.8)*.5*tis[i]*(Math.random()>.5?1:0):0, i=>Math.random()>.85&&tis[i]?.4*hem[i]:0,
    i=>eos[i]>.4&&hem[i]>.3?Math.random()*.3*tis[i]:0, i=>hem[i]>.55&&Math.random()>.8?.45*tis[i]:0,
    i=>hem[i]>.5&&eos[i]<.35?Math.random()*.4*tis[i]:0, i=>hem[i]>.45?Math.random()*.5*Math.pow(hem[i],1.3)*tis[i]:0,
    i=>hem[i]>.5&&Math.random()>.6?.55*tis[i]:0, i=>eos[i]>.45&&Math.random()>.75?.35*tis[i]:0,
    i=>eos[i]>.5?Math.pow(eos[i],1.5)*.6*tis[i]:0, i=>hem[i]>.6&&Math.random()>.8?.6*tis[i]:0,
    i=>eos[i]>.55&&Math.random()>.85?.4*tis[i]:0, i=>eos[i]>.35&&hem[i]<.35?eos[i]*.5*tis[i]:0,
    i=>Math.random()>.9&&tis[i]?.5*(.3*r[i]+.59*g[i]+.11*b[i]):0, i=>hem[i]>.65&&Math.random()>.9?.7*tis[i]:0,
    i=>eos[i]>.4&&hem[i]<.3?eos[i]*.4*tis[i]:0,
  ];
  const out=[];
  for(let ch=0;ch<21;ch++){
    const d=new Float32Array(width*height);
    for(let i=0;i<width*height;i++) d[i]=Math.max(0,Math.min(1,fns[ch](i)));
    const bl=new Float32Array(width*height);
    for(let y=1;y<height-1;y++) for(let x=1;x<width-1;x++){
      const idx=y*width+x;
      bl[idx]=d[idx]*.4+d[idx-1]*.1+d[idx+1]*.1+d[idx-width]*.1+d[idx+width]*.1+d[idx-width-1]*.05+d[idx-width+1]*.05+d[idx+width-1]*.05+d[idx+width+1]*.05;
    }
    out.push(bl);
  }
  return out;
}

function realToArrays(grids,dim){return grids.map(g=>{const a=new Float32Array(dim*dim);for(let y=0;y<dim;y++)for(let x=0;x<dim;x++)a[y*dim+x]=g[y][x]/255;return a;});}

function buildScene(ctr,imgUrl,chData,dim,vis,sp,el){
  const w=ctr.clientWidth,h=ctr.clientHeight;
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x08080f);scene.fog=new THREE.FogExp2(0x08080f,.12);
  const camera=new THREE.PerspectiveCamera(42,w/h,.1,100);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(w,h);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.3;
  ctr.innerHTML="";ctr.appendChild(renderer.domElement);
  scene.add(new THREE.AmbientLight(0x334466,.5));
  const dl=new THREE.DirectionalLight(0xffffff,1);dl.position.set(3,5,3);scene.add(dl);
  const pl=new THREE.PointLight(0x4488ff,.35,12);pl.position.set(-2,4,-1);scene.add(pl);
  const grp=new THREE.Group();scene.add(grp);
  const tex=new THREE.TextureLoader().load(imgUrl);tex.colorSpace=THREE.SRGBColorSpace;
  const base=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.MeshStandardMaterial({map:tex,side:THREE.DoubleSide,roughness:.55,metalness:.08}));
  base.rotation.x=-Math.PI/2;grp.add(base);
  const eg=new THREE.EdgesGeometry(new THREE.BoxGeometry(2.08,.015,2.08));
  grp.add(new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x445566})));
  function mkLbl(t,p){const c=document.createElement("canvas");c.width=256;c.height=64;const x=c.getContext("2d");x.fillStyle="#667788";x.font="bold 26px monospace";x.textAlign="center";x.fillText(t,128,40);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));s.position.copy(p);s.scale.set(.75,.19,1);grp.add(s);}
  mkLbl("x: H&E slide",new THREE.Vector3(0,-.14,1.28));mkLbl("y: mIF channels",new THREE.Vector3(1.28,-.14,0));
  const layers=[];
  for(let ch=0;ch<21;ch++){
    const data=chData[ch],col=new THREE.Color(CHANNELS[ch].color),pos=[],cls=[];
    for(let y=0;y<dim;y++)for(let x=0;x<dim;x++){
      const v=data[y*dim+x];if(v<.10)continue;
      pos.push((x/dim-.5)*2,(ch+1)*sp+v*el,(y/dim-.5)*2);
      const c=Math.min(v*1.6,1);cls.push(col.r*c,col.g*c,col.b*c);
    }
    if(!pos.length){layers.push(null);continue;}
    const gm=new THREE.BufferGeometry();gm.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));gm.setAttribute("color",new THREE.Float32BufferAttribute(cls,3));
    const pts=new THREE.Points(gm,new THREE.PointsMaterial({size:.022,vertexColors:true,transparent:true,opacity:.88,sizeAttenuation:true,blending:THREE.AdditiveBlending,depthWrite:false}));
    pts.visible=vis[ch];grp.add(pts);layers.push(pts);
  }
  let drag=false,prev={x:0,y:0},sph={theta:Math.PI/4.2,phi:Math.PI/4.5,r:3.6},_ar=true,t=0;
  function cu(){camera.position.set(sph.r*Math.sin(sph.phi)*Math.cos(sph.theta),sph.r*Math.cos(sph.phi),sph.r*Math.sin(sph.phi)*Math.sin(sph.theta));camera.lookAt(0,.55,0);}cu();
  const cv=renderer.domElement;
  cv.addEventListener("pointerdown",e=>{drag=true;prev={x:e.clientX,y:e.clientY};cv.setPointerCapture(e.pointerId);});
  cv.addEventListener("pointermove",e=>{if(!drag)return;sph.theta-=(e.clientX-prev.x)*.007;sph.phi=Math.max(.18,Math.min(1.5,sph.phi+(e.clientY-prev.y)*.007));prev={x:e.clientX,y:e.clientY};cu();});
  cv.addEventListener("pointerup",()=>{drag=false;});
  cv.addEventListener("wheel",e=>{e.preventDefault();sph.r=Math.max(1.4,Math.min(8,sph.r+e.deltaY*.004));cu();},{passive:false});
  let aid;function loop(){aid=requestAnimationFrame(loop);if(_ar&&!drag){t+=.0025;sph.theta=Math.PI/4.2+Math.sin(t)*.45;cu();}renderer.render(scene,camera);}loop();
  return{layers,setAR:v=>{_ar=v;},updVis:vs=>layers.forEach((l,i)=>{if(l)l.visible=vs[i];}),
    resize:()=>{const nw=ctr.clientWidth,nh=ctr.clientHeight;camera.aspect=nw/nh;camera.updateProjectionMatrix();renderer.setSize(nw,nh);},
    dispose:()=>{cancelAnimationFrame(aid);renderer.dispose();}};
}

export default function App(){
  const cRef=useRef(null),sRef=useRef(null),fRef=useRef(null),f2Ref=useRef(null);
  const[loaded,setLoaded]=useState(false);
  const[vis,setVis]=useState(()=>CHANNELS.map(()=>true));
  const[ar,setAr]=useState(true);
  const[hov,setHov]=useState(null);
  const[sp,setSp]=useState(0.12);
  const[el,setEl]=useState(0.18);
  const[busy,setBusy]=useState(false);
  const[src,setSrc]=useState(null);
  const[url,setUrl]=useState("http://localhost:7860");
  const[err,setErr]=useState(null);
  const imgRef=useRef(null),cdRef=useRef(null),dimR=useRef(128);

  const rebuild=useCallback((v,s,e)=>{
    if(!cRef.current||!cdRef.current||!imgRef.current)return;
    if(sRef.current)sRef.current.dispose();
    sRef.current=buildScene(cRef.current,imgRef.current,cdRef.current,dimR.current,v,s,e);
    sRef.current.setAR(ar);
  },[ar]);

  const doLocal=useCallback(f=>{
    if(!f)return;setBusy(true);setErr(null);
    const u=URL.createObjectURL(f),img=new Image();
    img.onload=()=>{
      const mx=128,sc=Math.min(mx/img.width,mx/img.height,1),w=Math.floor(img.width*sc),h=Math.floor(img.height*sc);
      const oc=document.createElement("canvas");oc.width=w;oc.height=h;
      const ctx=oc.getContext("2d");ctx.drawImage(img,0,0,w,h);
      cdRef.current=simulateChannels(ctx.getImageData(0,0,w,h).data,w,h);dimR.current=w;imgRef.current=u;
      setSrc("simulated");setLoaded(true);setBusy(false);
    };img.src=u;
  },[]);

  const doModel=useCallback(async f=>{
    if(!f)return;setBusy(true);setErr(null);
    try{
      // Step 1: Upload file
      const fd=new FormData();fd.append("files",f);
      const upRes=await fetch(`${url}/upload`,{method:"POST",body:fd});
      if(!upRes.ok) throw new Error(`Upload failed: ${upRes.status}`);
      const upJson=await upRes.json();
      const fpath=upJson[0];
      // Step 2: Call predict
      const pRes=await fetch(`${url}/api/predict`,{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({data:[{path:fpath,orig_name:f.name,size:f.size,mime_type:f.type}]}),
      });
      if(!pRes.ok) throw new Error(`Predict failed: ${pRes.status}`);
      const result=await pRes.json();
      // result.data = [gallery, html_string]
      const html=result.data[1];
      const m=html.match(/const DATA = ({[\s\S]*?});/);
      if(!m) throw new Error("Could not parse model response");
      const payload=JSON.parse(m[1]);
      cdRef.current=realToArrays(payload.channels,payload.grid_dim);
      dimR.current=payload.grid_dim;imgRef.current=payload.he_data_url;
      setSrc("model");setLoaded(true);setBusy(false);
    }catch(e){
      console.error(e);setErr(e.message);
      // fallback
      doLocal(f);
    }
  },[url,doLocal]);

  useEffect(()=>{if(loaded)rebuild(vis,sp,el);},[loaded,sp,el,rebuild]);
  useEffect(()=>{if(sRef.current)sRef.current.updVis(vis);},[vis]);
  useEffect(()=>{if(sRef.current)sRef.current.setAR(ar);},[ar]);
  useEffect(()=>{const fn=()=>{if(sRef.current)sRef.current.resize();};window.addEventListener("resize",fn);return()=>window.removeEventListener("resize",fn);},[]);
  useEffect(()=>()=>{if(sRef.current)sRef.current.dispose();},[]);

  const togCh=i=>setVis(v=>v.map((x,j)=>j===i?!x:x));
  const togAll=()=>{const a=vis.every(Boolean);setVis(CHANNELS.map(()=>!a));};

  return(
    <div style={{width:"100%",height:"100vh",display:"flex",fontFamily:"'JetBrains Mono','Fira Code',monospace",background:"linear-gradient(135deg,#08080f,#0d1117)",color:"#c8d6e5",overflow:"hidden"}}>
      <div style={{width:280,minWidth:280,height:"100%",overflowY:"auto",background:"rgba(12,15,22,.97)",borderRight:"1px solid rgba(80,120,200,.12)",display:"flex",flexDirection:"column"}}>
        {/* Header */}
        <div style={{padding:"12px 14px",borderBottom:"1px solid rgba(80,120,200,.08)",background:"linear-gradient(180deg,rgba(65,105,225,.06),transparent)"}}>
          <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:4}}>
            <div style={{width:26,height:26,borderRadius:6,display:"flex",alignItems:"center",justifyContent:"center",background:"linear-gradient(135deg,#4169E1,#00BCD4)",fontSize:13}}>🔬</div>
            <div><div style={{fontSize:12.5,fontWeight:700,color:"#e8f0fe"}}>GigaTIME</div>
              <div style={{fontSize:8,color:src==="model"?"#4a5":"#886",letterSpacing:"1.4px",textTransform:"uppercase"}}>{src==="model"?"Real Model · 3D":"3D mIF Viewer"}</div></div>
          </div>
          <div style={{fontSize:8,color:"#3a4556",lineHeight:1.4}}>21 virtual mIF channels · Research only</div>
        </div>

        {/* Backend URL */}
        <div style={{padding:"10px 14px",borderBottom:"1px solid rgba(80,120,200,.08)"}}>
          <div style={{fontSize:8,color:"#556",marginBottom:2}}>Gradio backend URL (gigatime_3d_integrated.py)</div>
          <input value={url} onChange={e=>setUrl(e.target.value)} style={{width:"100%",padding:"5px 8px",background:"rgba(255,255,255,.03)",border:"1px solid rgba(80,120,200,.15)",borderRadius:5,color:"#8899aa",fontSize:9,fontFamily:"inherit",outline:"none"}}/>
          <input ref={fRef} type="file" accept="image/*" style={{display:"none"}} onChange={e=>{if(e.target.files[0])doModel(e.target.files[0]);}}/>
          <button onClick={()=>fRef.current?.click()} style={{width:"100%",marginTop:6,padding:"9px 0",borderRadius:7,cursor:"pointer",fontFamily:"inherit",fontSize:10.5,fontWeight:600,background:"rgba(65,105,225,.08)",border:"1px dashed rgba(65,105,225,.35)",color:"#7aa2f7",transition:"all .2s"}}>
            {busy?"⏳ Running GigaTIME…":"⚡ Upload → Model Inference"}
          </button>
          <input ref={f2Ref} type="file" accept="image/*" style={{display:"none"}} onChange={e=>{if(e.target.files[0])doLocal(e.target.files[0]);}}/>
          <button onClick={()=>f2Ref.current?.click()} style={{width:"100%",marginTop:4,padding:"6px 0",borderRadius:7,cursor:"pointer",fontFamily:"inherit",fontSize:9,background:"transparent",border:"1px solid rgba(100,140,200,.12)",color:"#556"}}>
            📁 Simulated mode (no backend)
          </button>
          {err&&<div style={{marginTop:4,fontSize:8,color:"#f44",lineHeight:1.3}}>⚠ {err} — fell back to simulated</div>}
        </div>

        {/* Source badge */}
        {src&&<div style={{padding:"6px 14px",borderBottom:"1px solid rgba(80,120,200,.08)"}}>
          <div style={{display:"inline-flex",alignItems:"center",gap:5,padding:"3px 8px",borderRadius:4,
            background:src==="model"?"rgba(0,200,80,.08)":"rgba(200,150,0,.08)",
            border:`1px solid ${src==="model"?"rgba(0,200,80,.2)":"rgba(200,150,0,.2)"}`}}>
            <div style={{width:6,height:6,borderRadius:"50%",background:src==="model"?"#0c8":"#a80"}}/>
            <span style={{fontSize:8.5,color:src==="model"?"#0c8":"#a80",fontWeight:600}}>{src==="model"?"REAL MODEL PREDICTIONS":"SIMULATED DATA"}</span>
          </div></div>}

        {/* Controls */}
        <div style={{padding:"10px 14px",borderBottom:"1px solid rgba(80,120,200,.08)"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
            <span style={{fontSize:9,textTransform:"uppercase",letterSpacing:"1.2px",color:"#556"}}>Controls</span>
            <label style={{fontSize:10,color:"#7aa2f7",cursor:"pointer",display:"flex",alignItems:"center",gap:4}}>
              <input type="checkbox" checked={ar} onChange={e=>setAr(e.target.checked)} style={{accentColor:"#4169E1"}}/>Auto-rotate</label>
          </div>
          {[["Layer spacing",sp,setSp,.03,.35],["Height scale",el,setEl,.02,.6]].map(([l,v,s,mn,mx])=>(
            <div key={l} style={{marginBottom:5}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:8.5,color:"#556",marginBottom:1}}><span>{l}</span><span>{v.toFixed(2)}</span></div>
              <input type="range" min={mn} max={mx} step="0.005" value={v} onChange={e=>s(parseFloat(e.target.value))} style={{width:"100%",accentColor:"#4169E1",height:3}}/>
            </div>))}
        </div>

        {/* Channel list */}
        <div style={{padding:"6px 14px 3px",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={{fontSize:9,textTransform:"uppercase",letterSpacing:"1.2px",color:"#556"}}>Channels ({vis.filter(Boolean).length}/21)</span>
          <span onClick={togAll} style={{fontSize:9,color:"#7aa2f7",cursor:"pointer"}}>{vis.every(Boolean)?"Hide all":"Show all"}</span>
        </div>
        <div style={{flex:1,overflowY:"auto",padding:"0 10px 10px"}}>
          {CHANNELS.map((ch,i)=>(
            <div key={ch.name} onClick={()=>togCh(i)} onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}
              style={{display:"flex",alignItems:"center",gap:7,padding:"5px 6px",borderRadius:5,cursor:"pointer",marginBottom:1,transition:"all .12s",
                background:hov===i?"rgba(65,105,225,.08)":"transparent",opacity:vis[i]?1:.28}}>
              <div style={{width:9,height:9,borderRadius:3,flexShrink:0,background:ch.color,boxShadow:vis[i]?`0 0 5px ${ch.color}60`:"none"}}/>
              <div style={{flex:1,minWidth:0}}>
                <div style={{fontSize:10.5,fontWeight:600,color:vis[i]?"#e0e8f4":"#445"}}>{ch.name}</div>
                <div style={{fontSize:7.5,color:"#445",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{ch.desc}</div>
              </div>
              <div style={{width:5,height:5,borderRadius:"50%",background:vis[i]?ch.color:"rgba(80,120,200,.18)"}}/>
            </div>))}
        </div>
        <div style={{padding:"8px 14px",borderTop:"1px solid rgba(80,120,200,.08)",fontSize:7.5,color:"#2a3546",textAlign:"center",lineHeight:1.5}}>GigaTIME · Microsoft/Providence/UW · Cell 2025</div>
      </div>

      {/* 3D Canvas */}
      <div style={{flex:1,position:"relative"}}>
        <div ref={cRef} style={{width:"100%",height:"100%"}}>
          {!loaded&&<div style={{width:"100%",height:"100%",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:14}}>
            <div style={{width:110,height:110,borderRadius:22,background:"linear-gradient(135deg,rgba(65,105,225,.08),rgba(0,188,212,.08))",border:"1px solid rgba(65,105,225,.15)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:44}}>🧬</div>
            <div style={{textAlign:"center"}}>
              <div style={{fontSize:15,fontWeight:700,color:"#e8f0fe",marginBottom:5}}>GigaTIME 3D Viewer</div>
              <div style={{fontSize:10.5,color:"#667",maxWidth:340,lineHeight:1.6}}>
                <b style={{color:"#7aa2f7"}}>⚡ Model Inference</b> — sends your H&E tile to the Gradio backend running GigaTIME<br/><br/>
                <b style={{color:"#a80"}}>📁 Simulated</b> — derives approximate channel data from pixel colors (no GPU needed)
              </div>
            </div>
          </div>}
        </div>
        {loaded&&<div style={{position:"absolute",bottom:14,left:"50%",transform:"translateX(-50%)",background:"rgba(12,15,22,.82)",border:"1px solid rgba(80,120,200,.12)",borderRadius:7,padding:"5px 14px",fontSize:8.5,color:"#556",backdropFilter:"blur(6px)",pointerEvents:"none"}}>
          Drag to rotate · Scroll to zoom · Toggle channels in sidebar</div>}
      </div>
    </div>
  );
}
