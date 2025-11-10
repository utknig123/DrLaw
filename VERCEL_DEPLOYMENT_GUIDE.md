# Vercel Deployment Guide for DrLAW Frontend

This guide will help you deploy the frontend of your DrLAW application to Vercel.

## Prerequisites

1. A GitHub account
2. A Vercel account (sign up at https://vercel.com)
3. Your frontend code ready in the `frontend` folder
4. Your backend deployed on Render (or another hosting service)

## Step 1: Update Backend CORS Settings

Before deploying to Vercel, ensure your backend allows requests from your Vercel domain.

1. Go to your Render dashboard
2. Navigate to your backend service
3. Go to **Environment** tab
4. Add or update the `CORS_ORIGINS` environment variable:
   ```
   CORS_ORIGINS=https://your-vercel-app.vercel.app,http://localhost:3000
   ```
   (Replace `your-vercel-app` with your actual Vercel domain)

5. Save and restart your service

Alternatively, if you want to allow all origins (for testing), you can set:
```
CORS_ORIGINS=*
```

## Step 2: Update Frontend Configuration

1. Open `frontend/config.js`
2. Update the `BACKEND_URL` to match your Render backend URL:
   ```javascript
   const BACKEND_URL = 'https://your-backend.onrender.com';
   ```

## Step 3: Prepare Your Frontend for Vercel

### Option A: Deploy via GitHub (Recommended)

1. **Push your code to GitHub:**
   ```bash
   cd frontend
   git init
   git add .
   git commit -m "Initial commit for Vercel deployment"
   git branch -M main
   git remote add origin https://github.com/your-username/drlaw-frontend.git
   git push -u origin main
   ```

2. **Connect GitHub to Vercel:**
   - Go to https://vercel.com
   - Click **"Add New"** → **"Project"**
   - Click **"Import Git Repository"**
   - Select your GitHub repository
   - If this is your first time, authorize Vercel to access your GitHub

3. **Configure the Project:**
   - **Framework Preset:** Other
   - **Root Directory:** `./frontend` (or just `/` if the frontend folder is the root)
   - **Build Command:** (leave empty - static site)
   - **Output Directory:** (leave empty or set to `.`)
   - **Install Command:** (leave empty)

4. **Environment Variables (if needed):**
   - Usually not needed for static frontend, but you can add any if required

5. **Click "Deploy"**

### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Navigate to frontend folder:**
   ```bash
   cd frontend
   ```

3. **Login to Vercel:**
   ```bash
   vercel login
   ```

4. **Deploy:**
   ```bash
   vercel
   ```
   - Follow the prompts
   - For production deployment, use: `vercel --prod`

## Step 4: Configure Vercel Settings

1. **Go to your project settings on Vercel**

2. **Update vercel.json (if needed):**
   Your `vercel.json` should look like this:
   ```json
   {
     "version": 2,
     "routes": [
       {
         "src": "/",
         "dest": "/index.html"
       },
       {
         "src": "/login.html",
         "dest": "/login.html"
       },
       {
         "src": "/front.html",
         "dest": "/front.html"
       },
       {
         "src": "/(.*)",
         "dest": "/$1"
       }
     ]
   }
   ```

3. **Set up custom domain (optional):**
   - Go to **Settings** → **Domains**
   - Add your custom domain
   - Follow Vercel's instructions to configure DNS

## Step 5: Update Backend with Vercel URL

After deploying to Vercel, you'll get a URL like `https://your-app.vercel.app`

1. **Update Render Environment Variables:**
   - Go to your Render backend service
   - Navigate to **Environment** tab
   - Update `CORS_ORIGINS` to include your Vercel URL:
     ```
     CORS_ORIGINS=https://your-app.vercel.app,https://your-app-git-main.vercel.app,http://localhost:3000
     ```
   - Update `FRONTEND_URL` (if used):
     ```
     FRONTEND_URL=https://your-app.vercel.app
     ```



2. **Restart your Render service**

## Step 6: Test Your Deployment

1. Visit your Vercel URL: `https://your-app.vercel.app`
2. Test the following:
   - ✅ Sign up functionality
   - ✅ Login functionality
   - ✅ Logout functionality
   - ✅ Chat functionality
   - ✅ Google OAuth (if configured)

## Troubleshooting

### Issue: CORS Errors

**Solution:**
- Verify `CORS_ORIGINS` in Render includes your Vercel URL
- Check browser console for exact CORS error
- Ensure credentials are being sent: `credentials: 'include'` in fetch requests

### Issue: Logout Not Working

**Solution:**
- Check browser console for errors
- Verify the logout endpoint is being called correctly
- Ensure cookies are being sent with requests (`credentials: 'include'`)

### Issue: Session Not Persisting

**Solution:**
- Vercel uses HTTPS, so ensure your backend also uses HTTPS
- Check that cookies are being set with proper SameSite attributes
- Verify session cookie settings in Flask

### Issue: Images Not Loading

**Solution:**
- Ensure image paths are relative (e.g., `Logo.jpg` not `/static/Logo.jpg`)
- Check that all image files are in the `frontend` folder
- Verify file names match exactly (case-sensitive)

### Issue: 404 Errors on Refresh

**Solution:**
- Update `vercel.json` with proper routing rules
- Ensure all routes are handled correctly

## Important Notes

1. **Cookies and Sessions:**
   - Since frontend and backend are on different domains, ensure CORS is properly configured
   - Cookies need to be set with `SameSite=None; Secure` for cross-domain requests
   - Verify your Flask session settings support cross-origin requests

2. **Environment Variables:**
   - Frontend: Update `config.js` with your backend URL
   - Backend: Update `CORS_ORIGINS` and `FRONTEND_URL` environment variables

3. **Google OAuth:**
   - Update Google OAuth redirect URIs to include your Vercel URL
   - Update authorized JavaScript origins in Google Cloud Console

4. **Automatic Deployments:**
   - Vercel automatically deploys when you push to your connected GitHub repository
   - Preview deployments are created for pull requests

## Next Steps

1. Set up a custom domain (optional)
2. Configure environment variables for different environments (production, preview)
3. Set up monitoring and analytics
4. Optimize performance (image optimization, caching, etc.)

## Support

If you encounter issues:
1. Check Vercel deployment logs
2. Check browser console for errors
3. Verify backend logs on Render
4. Test API endpoints directly using Postman or curl

---

**Deployment Complete!** 🎉
Your frontend should now be live on Vercel and connected to your Render backend.









/



