# FaceSense  AI Wellness Monitor

## Project Structure

```
facesense-phase4/
├── app/
│   ├── layout.tsx             
│   ├── page.tsx               
│   ├── dashboard/
│   │   └── page.tsx           
│   └── globals.css
├── public/
│   ├── sw.js                  
│   └── models/                
├── server/
│   ├── index.js               
│   ├── config/db.js
│   ├── models/
│   │   ├── Session.js          
│   │   ├── Baseline.js         
│   │   ├── UserModel.js        
│   │   └── Alert.js            
│   └── routes/
│       ├── sessionRoutes.js    
│       ├── baselineRoutes.js   
│       ├── userModelRoutes.js  
│       ├── analyticsRoutes.js  
│       └── alertRoutes.js      
├── vercel.json                 
├── render.yaml                 
├── package.json                
├── .env.local.example
└── server/.env.example
```

---

## Quick Start (Local)

### 1. Frontend

```bash
npm install
npm run dev
```

### 2. Backend

```bash
cd server
npm install
node index.js
```
